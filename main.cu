#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_WIDTH 16
#define MAX_WIDTH 1024
#define POOL_SIZE 2
// matrix tile set to 25 to make offset calculation easier
#define CONV_TILE 25

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

// GPU functions
__global__ void convLayerForwardBasicKernel(float * X, float * W, float * Y, int W_grid, int input_wid, int output_wid, int mask_wid, int numInput, int numOutput) {
	int output_num = blockIdx.y;
	int input_num = blockIdx.x;
	int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;    //h tile index
	int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;     // w tiles index
	if ((h < output_wid) && (w < output_wid)){
		float acc = 0.0f;
		for (int c = 0; c < numInput; c++) {          // input features
			for (int p = 0; p < mask_wid; p++) {     //index in tile  height in ouput feature
				for (int q = 0; q < mask_wid; q ++) {    //index in tile width in ouput feature
					acc += (X[((input_num * input_wid + (h + p)) * input_wid + (w + q)) * numInput + c] *
					W[((p * mask_wid + q) * numInput + c) * numOutput  + output_num]);
				}
			}
		}
		Y[((input_num * output_wid + h) * output_wid + w) * numOutput + output_num] = acc;
	}
}

// This function uses matrix multiply to calculate convolution, it transform the input matrix
// and load that to shared memory on the fly
// We also combined relu to the writeback part of kernel
__global__ void convLayerForwardMatrixKernel(float * X, float * W, float * Y, int input_wid, int output_wid, int numInput, int numOutput) {
	__shared__ float W_tile[CONV_TILE][25];
	__shared__ float X_tile[25][CONV_TILE];
	int bx = blockIdx.x; // test case number
	int by = blockIdx.y; // output feature map number (by*CONV_TILE+ty)
	int bz = blockIdx.z; // output feature map dimension (bz*CONV_TILE+tz)
	int tx=threadIdx.x;  // output feature map number
	int ty=threadIdx.y;  // output feature map dimension
	// if tx and ty are within range
	int i,j;
	float sum=0.0f;
	int w_index,x_index;
	// offset doesn't change during iteration so calculate them in advance
	int x_offset=(bx*input_wid*input_wid+((bz*CONV_TILE+ty)/output_wid+tx/5)*input_wid+((bz*CONV_TILE+ty)%output_wid)+tx%5)*numInput;
	int w_offset=ty*numInput*numOutput+by*CONV_TILE+tx;
	for (i=0;i<numInput;i++){
		// get indexes
		w_index=w_offset+i*numOutput;
		x_index=x_offset+i;
		__syncthreads();
		if ((tx+by*CONV_TILE)<numOutput)
			W_tile[tx][ty]=W[w_index];
		else
			W_tile[tx][ty]=0.0f;
		if ((ty+bz*CONV_TILE)<(output_wid*output_wid))
			X_tile[tx][ty]=X[x_index];
		else
			X_tile[tx][ty]=0.0f;
		__syncthreads();
		// matrix multiply
		for (j=0;j<CONV_TILE;j++){
			sum+=(W_tile[tx][j]*X_tile[j][ty]);
		}
	}
	__syncthreads();
		// output format is the same as input, not the expanded matrix format
	if (((tx+by*CONV_TILE)<numOutput)&&(((ty+bz*CONV_TILE)/output_wid)<output_wid)){
		int y_offset=(bx*output_wid*output_wid+bz*CONV_TILE+ty)*numOutput+(tx+by*CONV_TILE);
		if (sum<0.0)
			sum=0.0f;
		Y[y_offset]=sum;
	}
}

__global__ void averagePool (float * X, float * Y, int W_grid, int input_wid, int output_wid, int numInput) {
	int output_num = blockIdx.y;
	int input_num = blockIdx.x;
	int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;    //h tile index
	int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;     // w tiles index
	if ((h < output_wid) && (w < output_wid)){
		float sum=0.0f;
		int yoffset = ((input_num * output_wid + h) * output_wid + w) * numInput + output_num;
		for (int p = 0; p < POOL_SIZE; p++) {     //index in tile  height   in ouput feature
			for (int q = 0; q < POOL_SIZE; q ++) {    //index in tile width in ouput feature
	      sum += X[((input_num * input_wid + POOL_SIZE * h + p)* input_wid + POOL_SIZE * w + q) * numInput + output_num];
			}
		}
		Y[yoffset]=sum/4.0;
	}
}

__global__ void gpu_relu4 (float * X, int total) {
	int X_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (X_idx < total){
		if (X[X_idx]<0.0)
			X[X_idx]=0.0f;
		// X[X_idx] = (X[X_idx] < 0) ? 0 : X[X_idx];
	}
}

// NN using shared memory
__global__ void gpu_fully_forward(float *X, float *W, float *Y, int output_size, int input_size){
	__shared__ float datain[1024];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	float tmp;
	// load data in
	// eliminate relu2 by checking if input is less than 0
	if (tx<input_size)
		tmp=X[bx*blockDim.x+tx];
		if (tmp<0)
			tmp=0.0f;
		datain[tx]=tmp;
	__syncthreads();
	// calculate result
	if (tx<output_size){
		int i;
		float sum=0.0f;
		for (i=0;i<input_size;i++)
			sum+=datain[i]*W[i*output_size+tx];
		Y[bx*output_size+tx]=sum;
	}
}



static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// From book chapter Figure 16.4
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];

  for (const auto i : range(0, ydims[0])) { //number of input feature maps
    for (const auto m : range(0, ydims[3])) { // number of output feature maps
      for (const auto h : range(0, ydims[1])) { // image width
        for (const auto w : range(0, ydims[2])) { // image height
          for (const auto p : range(0, filter_h)) { // filter height
            for (const auto q : range(0, filter_w)) { // filter width
              for (const auto c : range(0, in_channel)) { // number of filters
                const auto yoffset =
                    ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                     (h + p) * xdims[2] * xdims[3] +
                                     (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                                     q * wdims[2] * wdims[3] + c * wdims[3] + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]) {
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset =
                  ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                   (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}

static void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  for (const auto i : range(0, xdims[0])) {
    for (const auto j : range(0, wdims[1])) {
      float sum = 0;
      for (const auto k : range(0, xdims[1])) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

	// int i,j,k,l;
  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);
	//
	// for (i=0;i<10;i++){
	// 	for (j=0;j<24;j++){
	// 		for (k=0;k<24;k++){
	// 			for (l=0;l<32;l++)
	// 				printf("%.4f ",a[i*32*24*24+j*32*24+k*32+l]);
	// 				printf("\n");
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n\n");
	// }

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // relu
  relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);


  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);
	// for (i=0;i<10;i++){
	// 	for (j=0;j<10;j++){
	// 		printf("%.4f ",f[i*10+j]);
	// 	}
	// 	printf("\n");
	// }

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation_gpu(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  float *conv1_input;
  float *conv1_output;
  float *conv2_input;
  float *conv2_output;
  float *W1;
  float *W2;
  float *NN_L1_input;
	float *NN_L2_input;
	float *NN_output_gpu;
	float *NN_L1_weights;
	float *NN_L2_weights;
	int argdim[2]={xdims[0],fc2dims[1]};
	float *argmax_input=zeros<float>(argdim);
  int x1dim[4]={xdims[0],xdims[1],xdims[2],xdims[3]};
  int y1dim[4]={xdims[0],xdims[1]-conv1dims[0]+1,xdims[2]-conv1dims[1]+1,conv1dims[3]};
  int x2dim[4]={xdims[0],y1dim[1]/2,y1dim[2]/2,y1dim[3]};
  int y2dim[4]={xdims[0],x2dim[1]-conv2dims[0]+1,x2dim[2]-conv2dims[1]+1,conv2dims[3]};
  int NN_1_dim[4]={xdims[0],y2dim[1]/2,y2dim[2]/2,y2dim[3]};
	int NN_2_dim[2]={xdims[0],fc1dims[1]};
	// allocate global memory
  check_success(cudaMalloc(&conv1_input,sizeof(float)*x1dim[0]*x1dim[1]*x1dim[2]*x1dim[3]));
  check_success(cudaMalloc(&conv1_output,sizeof(float)*y1dim[0]*y1dim[1]*y1dim[2]*y1dim[3]));
  check_success(cudaMalloc(&conv2_input,sizeof(float)*x2dim[0]*x2dim[1]*x2dim[2]*x2dim[3]));
  check_success(cudaMalloc(&conv2_output,sizeof(float)*y2dim[0]*y2dim[1]*y2dim[2]*y2dim[3]));
  check_success(cudaMalloc(&W1,sizeof(float)*conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3]));
  check_success(cudaMalloc(&W2,sizeof(float)*conv2dims[0]*conv2dims[1]*conv2dims[2]*conv2dims[3]));
  check_success(cudaMalloc(&NN_L1_input,sizeof(float)*NN_1_dim[0]*NN_1_dim[1]*NN_1_dim[2]*NN_1_dim[3]));
  check_success(cudaMalloc(&NN_L2_input,sizeof(float)*NN_2_dim[0]*NN_2_dim[1]));
  check_success(cudaMalloc(&NN_L1_weights,sizeof(float)*fc1dims[0]*fc1dims[1]));
  check_success(cudaMalloc(&NN_L2_weights,sizeof(float)*fc2dims[0]*fc2dims[1]));
  check_success(cudaMalloc(&NN_output_gpu,sizeof(float)*argdim[0]*argdim[1]));
  check_success(cudaMemcpy(conv1_input,x,sizeof(float)*x1dim[0]*x1dim[1]*x1dim[2]*x1dim[3],cudaMemcpyHostToDevice));
  check_success(cudaMemcpy(W2,conv2,sizeof(float)*conv2dims[0]*conv2dims[1]*conv2dims[2]*conv2dims[3],cudaMemcpyHostToDevice));
	check_success(cudaMemcpy(W1,conv1,sizeof(float)*conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3],cudaMemcpyHostToDevice));
	check_success(cudaMemcpy(NN_L1_weights,fc1,sizeof(float)*fc1dims[0]*fc1dims[1],cudaMemcpyHostToDevice));
	check_success(cudaMemcpy(NN_L2_weights,fc2,sizeof(float)*fc2dims[0]*fc2dims[1],cudaMemcpyHostToDevice));
  int Y=y1dim[1]*y1dim[2]/CONV_TILE;
	if ((y1dim[1]*y1dim[2])%CONV_TILE)
		Y++;
	int X=y1dim[3]/CONV_TILE;
	if (y1dim[3]%CONV_TILE)
		X++;
	int W_grid=0;
  dim3 conv1_block (CONV_TILE, CONV_TILE,1);
  dim3 conv1_grid (xdims[0],X,Y);
  convLayerForwardMatrixKernel<<<conv1_grid,conv1_block>>>(conv1_input,W1,conv1_output,x1dim[1],y1dim[1],x1dim[3],y1dim[3]);
  // average pool
  W_grid = x2dim[1] / TILE_WIDTH;
  int H_grid = x2dim[2] / TILE_WIDTH;
  if (y1dim[1]%TILE_WIDTH){
    W_grid++;
    H_grid++;
  }
  Y = H_grid * W_grid;
  dim3 avg_pool_1_block (TILE_WIDTH, TILE_WIDTH,1);
  dim3 avg_pool_1_grid (xdims[0], x2dim[3], Y);
  averagePool<<<avg_pool_1_grid,avg_pool_1_block>>>(conv1_output,conv2_input,W_grid,y1dim[1],x2dim[1],y1dim[3]);
	// second conv
  Y=y2dim[1]*y2dim[2]/CONV_TILE;
	if ((y2dim[1]*y2dim[2])%CONV_TILE)
		Y++;
	X=y2dim[3]/CONV_TILE;
	if (y2dim[3]%CONV_TILE)
		X++;
  dim3 conv2_block (CONV_TILE, CONV_TILE,1);
  dim3 conv2_grid (xdims[0],X,Y);
  convLayerForwardMatrixKernel<<<conv2_grid,conv2_block>>>(conv2_input,W2,conv2_output,x2dim[1],y2dim[1],x2dim[3],y2dim[3]);
	// average pooling
  W_grid = NN_1_dim[1] / TILE_WIDTH;
  H_grid = NN_1_dim[2] / TILE_WIDTH;
  if (y1dim[1]%TILE_WIDTH){
    W_grid++;
    H_grid++;
  }
  Y = H_grid * W_grid;
  dim3 avg_pool_2_block (TILE_WIDTH, TILE_WIDTH,1);
  dim3 avg_pool_2_grid (xdims[0], NN_1_dim[3], Y);
  averagePool<<<avg_pool_2_grid,avg_pool_2_block>>>(conv2_output,NN_L1_input,W_grid,y2dim[1],NN_1_dim[1],y2dim[3]);
	dim3 fully_forward_1_grid(xdims[0],1,1);
	dim3 fully_forward_1_block(fc1dims[0],1,1);
	gpu_fully_forward<<<fully_forward_1_grid,fully_forward_1_block>>>(NN_L1_input,NN_L1_weights,NN_L2_input,fc1dims[1],fc1dims[0]);
	dim3 fully_forward_2_grid(xdims[0],1,1);
	dim3 fully_forward_2_block(fc2dims[0],1,1);
	gpu_fully_forward<<<fully_forward_2_grid,fully_forward_2_block>>>(NN_L2_input,NN_L2_weights,NN_output_gpu,fc2dims[1],fc2dims[0]);
  check_success(cudaMemcpy(argmax_input,NN_output_gpu,sizeof(float)*xdims[0]*fc2dims[1],cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
  check_success(cudaFree(conv1_input));
  check_success(cudaFree(conv1_output));
  check_success(cudaFree(conv2_input));
  check_success(cudaFree(conv2_output));
  check_success(cudaFree(W1));
  check_success(cudaFree(W2));
  check_success(cudaFree(NN_L1_input));
  check_success(cudaFree(NN_L2_input));
  check_success(cudaFree(NN_L1_weights));
  check_success(cudaFree(NN_L2_weights));
  check_success(cudaFree(NN_output_gpu));
  const int fdims[] = {xdims[0], fc2dims[1]};
  argmax(argmax_input, fdims, out);
	free(argmax_input);
}

int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation_gpu(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
	for (const auto i : range(0, FLAGS_batch_size)) {
		if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}



