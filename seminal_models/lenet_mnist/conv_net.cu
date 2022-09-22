#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// #pragma unroll

#define CHUNK_SIZE 64

#define SM_COUNT 82
#define WARP_PER_SM 4
#define THREAD_PER_WARP 32
#define MAX_THREAD_PER_BLOCK 1024



// matrix set zero is same as vector zero, (flatten to 1D)
__global__  void setZero(int size, float *A){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      A[i] = 0;
  }
}

// matrix set zero is same as vector zero, (flatten to 1D)
__global__  void setVal(int size, float *A, float val){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      A[i] = val;
  }
}

// size is the number of samples = number of columns (minibatch size)
// rowInd is the row value in each sample to set to null
__global__  void setRowVal(int width, int rowInd, float *A, float val){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width){
      A[width * rowInd + i] = val;  
  }
}

// size is the number of samples = number of columns (minibatch size)
// rowInd is the row value in each sample to set to null
__global__  void elementSquare(int size, float *A, float *out){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      out[i] = A[i] * A[i];  
  }
}

// size is the number of samples = number of columns (minibatch size)
// rowInd is the row value in each sample to set to null
__global__  void elementMult(int size, float *A, float *B, float *out){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      out[i] = A[i] * B[i];  
  }
}

// matrix add is same as vector add, flatten to 1D
__global__  void matAdd(int size, float *A, float *B, float *out){

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size){
      out[i] = A[i] + B[i];
  }
}


__global__  void matSub(int size, float *A, float *B, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      out[i] = A[i] - B[i];
  }
}


__global__  void matScale(int size, float *A, float factor){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      A[i] = A[i] * factor;
  }
}

__global__  void matScaleEls(int size, float *A, float *B){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      A[i] = A[i] * B[i];
  }
}


__global__  void computeNewSecondDeriv(int size, float *current, float *prior, float new_weighting){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      current[i] = (1 - new_weighting) * prior[i] + new_weighting * current[i];
  }
}


__global__  void computeStepSize(int size, float *A, float learning_rate, float safety_factor, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      out[i] = -learning_rate / (safety_factor + fabsf(A[i]));
      //out[i] = -learning_rate;
  }
}


// size is the batch size (number of values in loss array)
// output_len is the number of nodes per sample (10 in MNIST Case)
// X is the value of output nodes (output_len X batch size), Y is the correct labels (output_len X batch size)
__global__  void computeLoss(int size, int output_len, float *X, float *Y, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    float sum, diff;
    sum = 0;
    for (int j = 0; j < output_len; j++){
      diff = (Y[i + j * size] - X[i + j * size]);
      sum += (diff * diff);
    }
    out[i] = sum / output_len;
  }
}

__global__ void computeNorm(int size, float* gradient, float *out){
  out[0] = normf(size, gradient);
}

__global__ void gradientClip(int size, float *gradient, float threshold, float *norm){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size && norm[0] > threshold){
    gradient[i] = threshold * gradient[i] / norm[0];
  }
}


// for sigmoid: hadamard(gradient, (hadamard(nodes, 1 - nodes)))
// for tanh: 1 - f(x) ^ 2
// here "gradient" is upstream gradient being propogated backwards
// nodes is the W*x + B value at the layer before activation (A_layer variable)
__global__  void activationDeriv(int size, float *gradient, float *nodes, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      float TANH_FREQ = 2.0 / 3.0;
      float TANH_AMP = 1.7159;
      float tanh = tanhf(nodes[i]);
      float tanh_sq = tanh * tanh;
      //out[i] = gradient[i] * (1 - (nodes[i] * nodes[i])) * TANH_FREQ * TANH_AMP;
      out[i] = gradient[i] * (1 - tanh_sq);
  }
}

// for sigmoid: hadamard(gradient, (hadamard(nodes, 1 - nodes)))
// for tanh: 1 - f(x) ^ 2
__global__  void activationSecondDeriv(int size, float *upstreamSeconDeriv, float *nodes, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      float TANH_FREQ = 2.0 / 3.0;
      float TANH_AMP = 1.7159;
      // taylor series second deriv of activation function => 1 / 2! * (f'(x))^2
      // f(x) = tanh
      // -2f(x) * (1 - f(x)^2)
      float tanh = tanhf(nodes[i]);
      float tanh_sq = tanh * tanh;
      float deriv = 1 - tanh_sq;
      out[i] = upstreamSeconDeriv[i] * -2 * tanh * deriv;
  }
}

// [blocks of 100 values per sample sequentially where the 100 values are the softmax jacobian for that sample]
__global__ void softMaxJacobian(int size, int width, int output_len, float * softmax_values, float *out){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    int sample_num = i / (output_len * output_len);
    int sample_pos = i % (output_len * output_len);
    int rowInd = sample_pos / output_len;
    int colInd = sample_pos % output_len;
    float rowValue = softmax_values[sample_num + width * rowInd];
    float colValue = softmax_values[sample_num + width * colInd];
    // float rowValue = fmaxf(0.01, softmax_values[sample_num + width * rowInd]);
    // float colValue = fmaxf(0.01, softmax_values[sample_num + width * colInd]);
    if (rowInd == colInd){
      out[i] = rowValue * (1 - rowValue);
    }
    else{
      out[i] = -1 * rowValue * colValue;
    }
  }

}
// for sigmoid: hadamard(gradient, (hadamard(nodes, 1 - nodes)))
// 
__global__  void activationDerivSoft(int size, int width, int output_len, float *gradient, float *softmax_jacobian, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  float sum = 0;
  if (i < size){
      for (int k = 0; k < output_len; k++){
        sum += gradient[width * k + colInd] * softmax_jacobian[output_len * output_len * colInd + output_len * rowInd + k];
      }
      out[i] = sum;
  }
}

// for sigmoid: hadamard(gradient, (hadamard(nodes, 1 - nodes)))
// for tanh: 1 - f(x) ^ 2
__global__  void activationSecondDerivSoft(int size, float *upstreamSecondDeriv, float *softmax_values, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      out[i] = upstreamSecondDeriv[i] * softmax_values[i] * (1 - softmax_values[i]);
  }
}

// node derivs is from activation derivative
// size is number of nodes at layer; width is batch size
__global__  void biasDerivs(int size, int width, float *node_derivs, float *out){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
      float sum = 0;
      for (int j = 0; j < width; j++){
        sum += node_derivs[width * i + j];
      }
      out[i] = sum;
  }
}


// ASSUME BLOCK + THREAD ARE 1-D
// size is batch size
// one thread per sample in batch
__global__ void softMax(int size, int output_len, float*X){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    float sum = 0;
    for (int j = 0; j < output_len; j++){
      sum += __expf(X[i + size * j]);
      if (isnan(sum) && !isnan(X[i + size * j])){
        printf("Softmax Explosion - Z = %f\n", X[i + size * j]);
      }
    }
    for (int j = 0; j < output_len; j++){
      X[i + size * j] = __expf(X[i + size * j]) / sum;
    }

  }
}

// // ASSUME BLOCK + THREAD ARE 1-D
// // size is batch size
// // one thread per sample in batch
// __global__ void softMaxDeriv(int width, int output_len, int width, float*X, float*out){
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < size){
//     for (int i = 0; i < output_len; i++){

//     }
//   }
// }

// ASSUME BLOCK + THREAD ARE 1-D
// can work with vectors as well
// size is total number of elements (nRows * nCols), width = nCols
__global__ void transposeSimp(int size, int width, float * M, float *out){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  int height = size / width;
  if (i < size){
    out[height * colInd + rowInd] = M[width * rowInd + colInd];
  }
}


// ASSUME BLOCK + THREAD ARE 1-D
// can work with vectors as well
// size is total number of elements (nRows * nCols), width = nCols
__global__  void addBias(int size, int width, float *X, float *B){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  if (i < size){
    X[width * rowInd + colInd] = X[width * rowInd + colInd] + B[rowInd];
  }
}


// ASSUME BLOCK + THREAD ARE 1-D
// can work with vectors as well
// size is total number of elements (nRows * nCols), width = nCols
__global__  void activate(int size, int width, float *X){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / width;
  int colInd = i % width;
  float z;
  if (i < size){
    float TANH_FREQ = 2.0 / 3.0;
    float TANH_AMP = 1.7159;
    z = TANH_FREQ * X[width * rowInd + colInd];
    X[width * rowInd + colInd] = TANH_AMP * tanhf(z);
  }
}

__global__  void makePredict(int size, int output_len, float *X, float *out){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float pred_val = -1;
  float pred_ind = -1;
  float val;
  if (i < size){
    for (int j = 0; j < output_len; j++){
      val = X[i + j * size];
      if (val > pred_val) {
        pred_val = val;
        pred_ind = j;
      }
    }
    out[i] = pred_ind;
  }
}


// ASSUME BLOCK + THREAD ARE BOTH 1-D
// very un-optimized, but good for testing...
// zero out matrix before matrix multiply
// blockDim.x * blockIdx.x + threadIdx.x represents index in 1-D array of AB
__global__  void matMulSimp(int M, int K, int N, float *A, float *B, float *out){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int rowInd = i / N;
  int colInd = i % N;
  if (i < M * N){
    out[rowInd * N + colInd] = 0;
    for (int i = 0; i < K; i++){
      out[rowInd * N + colInd] += A[rowInd * K + i] * B[i * N + colInd];
    }
  }
}


__global__ void fullWeightDerivToUniqueNonNull(int size, int *map_to_unique_ind, float *full_W,  float *out){
  int full_w_ind = blockDim.x * blockIdx.x + threadIdx.x;
  if (full_w_ind < size){
    int unique_ind = map_to_unique_ind[full_w_ind];
    if (unique_ind != -1){
      atomicAdd(&out[unique_ind], full_W[full_w_ind]);
    }
  }
}

__global__ void fullWeightDerivToUniqueNull(int size, int width, float *full_W,  float *out, int * null_input_mappings, int null_input_mappings_width, int * null_set_sizes, int null_ind){
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < size){
    float virt_null_weight_value = full_W[width * row + null_ind];
    float null_set_size = null_set_sizes[row];
    if (null_set_size > 0){
      //float deriv_portion = virt_null_weight_value / null_set_size;
      float deriv_portion = virt_null_weight_value;
      int unique_ind;
      for (int i = 0; i < null_set_size; i++){
        unique_ind = null_input_mappings[null_input_mappings_width * row + i];
        atomicAdd(&out[unique_ind], deriv_portion);
      }
    }
  }
}




__global__ void repopulate_with_unique_non_null_gpu(int size, float * full_W, float *unique_W, int unique_W_len, int * map_from_unique, int map_from_unique_width){
  int from_unique_ind = blockDim.x * blockIdx.x + threadIdx.x;
  if (from_unique_ind < size){
    int weight_ind = from_unique_ind / map_from_unique_width;
    int full_w_ind = map_from_unique[from_unique_ind];
    if (full_w_ind != -1){
      full_W[full_w_ind] = unique_W[weight_ind];
    }
  }
}

__global__ void repopulate_with_unique_null_gpu(int size, float * full_W, float *unique_W, int unique_W_len, int * null_input_mappings, int null_input_mappings_width, int null_ind, int full_W_width){
  // for each output row, populate the column corresponding to null input with the sum of all unique inds in the null space
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  int weight_ind;
  if (row < size){
    full_W[null_ind + row * full_W_width] = 0;
    for (int i = 0; i < null_input_mappings_width; i++){
        if (null_input_mappings[row * null_input_mappings_width + i] == -1){
          break;
      }
      weight_ind = null_input_mappings[row * null_input_mappings_width + i];
      sum += unique_W[weight_ind];
    }
    full_W[null_ind + row * full_W_width] = sum;
  }
} 



void add_conv_mappings(int input_map_num, int input_map_cnt_for_output, int k, int map_i, int map_j, int inp_N, int conv_N, int map_N, int n_maps_connected, int * map_to_unique, int map_to_unique_width, int * map_from_unique, int map_from_unique_width, int * null_input_mappings, int null_input_mappings_width, int *null_set_sizes){
  int input_focus_row = 2 * map_i;
  int input_focus_col = 2 * map_j;
  int mid = conv_N / 2;
  int input_connection_ind, output_connection_ind, weight_ind;
  for (int kern_i = -mid; kern_i <= mid; kern_i++){
    for (int kern_j = -mid; kern_j <= mid; kern_j++){
      output_connection_ind = k * (map_N * map_N) + map_N * map_i + map_j;
      weight_ind = k * (conv_N * conv_N) * n_maps_connected + input_map_cnt_for_output * (conv_N * conv_N) + conv_N * (kern_i + mid) + (kern_j + mid);
      if ((input_focus_row + kern_i < 0) || (input_focus_row + kern_i >= inp_N) || (input_focus_col + kern_j < 0) || (input_focus_col + kern_i >= inp_N)){
        for (int insert_ind = 0; insert_ind < null_input_mappings_width; insert_ind++){
          if (null_input_mappings[null_input_mappings_width * output_connection_ind + insert_ind] == -1) {
            // for a given output node, store the unique weights which take a null node as input
            null_input_mappings[null_input_mappings_width * output_connection_ind + insert_ind] = weight_ind;
            null_set_sizes[output_connection_ind] += 1;
            break;
          }
        }
      }
      else{
        input_connection_ind = input_map_num * (inp_N * inp_N) + inp_N * (input_focus_row + kern_i) + (input_focus_col + kern_j);
        map_to_unique[map_to_unique_width * output_connection_ind + input_connection_ind] = weight_ind;
        for (int insert_ind = 0; insert_ind < map_from_unique_width; insert_ind++){
          if (map_from_unique[map_from_unique_width * weight_ind + insert_ind] == -1){
            map_from_unique[map_from_unique_width * weight_ind + insert_ind] = map_to_unique_width * output_connection_ind + input_connection_ind;
            break;
          }
        }
      }
    }
  }
}



void repopulate_with_unique(float * full_W, float *unique_W, int unique_W_len, int * map_from_unique, int map_from_unique_width, int * null_input_mappings, int null_input_mappings_width, int null_ind, int full_W_rows, int full_W_width){
  int full_w_ind;
  for (int weight_ind = 0; weight_ind < unique_W_len; weight_ind++){
    for (int j = 0; j < map_from_unique_width; j++){
        full_w_ind = map_from_unique[map_from_unique_width * weight_ind + j];
        if (full_w_ind == -1){
          break;
        }
        full_W[full_w_ind] = unique_W[weight_ind];
    }
  }
  // for each output row, populate the column corresponding to null input with the sum of all unique inds in the null space
  int weight_ind;
  for (int row = 0; row < full_W_rows; row++){
    // before adding the weights corresponding to null input re-init to 0
    full_W[null_ind + row * full_W_width] = 0;
    for (int i = 0; i < null_input_mappings_width; i++){
      if (null_input_mappings[row * null_input_mappings_width + i] == -1){
        break;
      }
      weight_ind = null_input_mappings[row * null_input_mappings_width + i];
      full_W[null_ind + row * full_W_width] += unique_W[weight_ind];
    }
  }
}

void my_write_file(int size, float * data, const char * filename){
  FILE * fp = fopen(filename, "w+");
  fwrite(data, sizeof(float), size_t (size), fp);
  fclose(fp);
}


int main(void)
{


  // DEFINE ARCHITECURAL PARAMTERS FOR NEURAL NET 

  // mini batch size
  int batch_size = 1;
  // how many times to repeat dataset
  int repeat_n = 23;
  float learning_rate_sched[repeat_n];
  // resetting to learning rate...
  float safety_factor = 0.001;
  //float clipping_threshold = 2;
  float second_deriv_mem = 0.001;
  for (int i = 0; i < repeat_n; i++){
    if (i < 2){
      learning_rate_sched[i] = .0005;
    }
    else if (i < 5){
      learning_rate_sched[i] = .0001;
    }
    else if (i < 9){
      learning_rate_sched[i] = .00005;
    }
    else{
      learning_rate_sched[i] = .00001;
    }
  }

  // for (int i = 0; i < repeat_n; i++){
  //   learning_rate_sched[i] = .05; 
  // }

  int input_len = 257;
  int output_len = 10;

  int h1_size = 769;
  int h2_size = 192;
  int h3_size = 30;

  int input_dim = 16;
  int input_null_ind = 256;

  int h1_maps = 12;
  int h1_kernel_dim = 5;
  int h1_map_dim = 8;
  int h1_null_ind = 768;

  int h2_maps = 12;
  int h2_kernel_dim = 5;
  int h2_map_dim = 4;
  int h2_maps_connected_to_h1 = 8; 

  int W_h1_size = h1_maps * (h1_kernel_dim * h1_kernel_dim);
  int W_h1_full_size = input_len * h1_size;
  int W_h2_size = h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim);
  int W_h2_full_size = h1_size * h2_size;
  int W_h3_size = h2_size * h3_size;
  int W_out_size = h3_size * output_len;


  // input and labels
  float *X_in_host, *Y_out_host;
  X_in_host = (float*)malloc(input_len * batch_size *sizeof(float));
  Y_out_host = (float*)malloc(output_len * batch_size * sizeof(float));

  // for checking values...
  float *X_h1_host, *X_h2_host, *X_h3_host, *X_out_host;
  X_h1_host = (float*)malloc(h1_size * batch_size *sizeof(float));
  X_h2_host = (float*)malloc(h2_size * batch_size *sizeof(float));
  X_h3_host = (float*)malloc(h3_size * batch_size *sizeof(float));
  X_out_host = (float*)malloc(output_len * batch_size *sizeof(float));

  float *X_h3_T_host;
  X_h3_T_host = (float *)malloc(h3_size * batch_size * sizeof(float));

  float *dX_h1_host, *dX_h2_host, *dX_h3_host, *dX_out_host;
  dX_h1_host = (float*)malloc(h1_size * batch_size *sizeof(float));
  dX_h2_host = (float*)malloc(h2_size * batch_size *sizeof(float));
  dX_h3_host = (float*)malloc(h3_size * batch_size *sizeof(float));
  dX_out_host = (float*)malloc(output_len * batch_size *sizeof(float));

  float *dX_h1_activation_host, *dX_h2_activation_host, *dX_h3_activation_host, *dX_out_activation_host;
  dX_h1_activation_host = (float*)malloc(h1_size * batch_size *sizeof(float));
  dX_h2_activation_host = (float*)malloc(h2_size * batch_size *sizeof(float));
  dX_h3_activation_host = (float*)malloc(h3_size * batch_size *sizeof(float));
  dX_out_activation_host = (float*)malloc(output_len * batch_size *sizeof(float));



  // weights
  float *W_h1_host, *W_h2_host, *W_h3_host, *W_out_host;
  W_h1_host = (float *)malloc(W_h1_size * sizeof(float));
  W_h2_host = (float *)malloc(W_h2_size * sizeof(float));
  W_h3_host = (float *)malloc(W_h3_size * sizeof(float));
  W_out_host = (float *)malloc(W_out_size * sizeof(float));




  // for checking derivs...
  float *dW_h1_host, *dW_h2_host, *dW_h3_host, *dW_out_host;
  dW_h1_host = (float *)malloc(W_h1_size * sizeof(float));
  dW_h2_host = (float *)malloc(W_h2_size * sizeof(float));
  dW_h3_host = (float *)malloc(W_h3_size * sizeof(float));
  dW_out_host = (float *)malloc(W_out_size * sizeof(float));

  float *W_h1_full_host, *W_h2_full_host, *W_h2_full_T_host;
  W_h1_full_host = (float *)malloc(W_h1_full_size * sizeof(float));
  W_h2_full_host = (float *)malloc(W_h2_full_size * sizeof(float));
  W_h2_full_T_host = (float *)malloc(W_h2_full_size * sizeof(float));

  // biases
  float *B_h1_host, *B_h2_host, *B_h3_host, *B_out_host;
  B_h1_host = (float *)malloc(h1_size * sizeof(float));
  B_h2_host = (float *)malloc(h2_size * sizeof(float));
  B_h3_host = (float *)malloc(h3_size * sizeof(float));
  B_out_host = (float *)malloc(output_len * sizeof(float));

  // loss (storing loss values per batch)
  float *loss_host;
  loss_host = (float *)malloc(batch_size * sizeof(float));

  // predicted values
  float *predicted_host;
  predicted_host = (float*)malloc(batch_size *sizeof(float));


  // create mappings between indicies of duplicated weight matrix (full) and indicies unique weight vector 
  int *h1_null_mappings_host;
  int h1_null_mappings_width = (h1_kernel_dim * h1_kernel_dim);
  h1_null_mappings_host = (int *)malloc(h1_size * h1_null_mappings_width * sizeof(int));
  for (int i = 0; i < h1_size * h1_null_mappings_width; i++){
    h1_null_mappings_host[i] = -1;
  }

  int *h2_null_mappings_host;
  int h2_null_mappings_width = (h2_kernel_dim * h2_kernel_dim) * h2_maps_connected_to_h1;
  h2_null_mappings_host = (int *)malloc(h2_size * h2_null_mappings_width * sizeof(int));
  for (int i = 0; i < h2_size * h2_null_mappings_width; i++){
    h2_null_mappings_host[i] = -1;
  }

  // get sizes of number of unique weights which are coming from null node
  int *h1_null_set_sizes_host;
  h1_null_set_sizes_host = (int *)malloc(h1_size * sizeof(int));
  for (int i = 0; i < h1_size; i++){
    h1_null_set_sizes_host[i] = 0;
  }

  int *h2_null_set_sizes_host;
  h2_null_set_sizes_host = (int *)malloc(h2_size * sizeof(int));
  for (int i = 0; i < h2_size; i++){
    h2_null_set_sizes_host[i] = 0;
  }


  
  // array of h1_size rows and input_len columns
  int * h1_full_to_unique_host = (int *)malloc(input_len * h1_size * sizeof(int));
  for (int i = 0; i < input_len * h1_size; i++){
    h1_full_to_unique_host[i] = -1;
  }
  // array of kernel_maps * (kern_dim ** 2) rows with map_dim ** 2 columns 
  int * h1_unique_to_full_host = (int *)malloc((h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim) * sizeof(int));
  // initalize unique -> full with -1 to know insertion location for reverse mapping...
  for (int i = 0; i < h1_maps * (h1_kernel_dim * h1_kernel_dim) * (h1_map_dim * h1_map_dim); i++){
    h1_unique_to_full_host[i] = -1;
  }

  // array of h2_size rows and h1_size columns
  int * h2_full_to_unique_host = (int *)malloc(h1_size * h2_size * sizeof(int));
  for (int i = 0; i < h1_size * h2_size; i++){
    h2_full_to_unique_host[i] = -1;
  }
  int * h2_unique_to_full_host = (int *)malloc((h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim) * sizeof(int));
  for (int i = 0; i < h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim) * (h2_map_dim * h2_map_dim); i++){
    h2_unique_to_full_host[i] = -1;
  }

 
  // input to h1 mappings
  for (int k = 0; k < h1_maps; k++){
    for (int map_i = 0; map_i < h1_map_dim; map_i++){
      for (int map_j = 0; map_j < h1_map_dim; map_j++){
        add_conv_mappings(0, 0, k, map_i, map_j, input_dim, h1_kernel_dim, h1_map_dim, 1, h1_full_to_unique_host, input_len, h1_unique_to_full_host, h1_map_dim * h1_map_dim, h1_null_mappings_host, h1_null_mappings_width, h1_null_set_sizes_host);
      }
    }
  }

  // h1 to h2 mappings
  for (int k = 0; k < h2_maps; k++){
    for (int map_i = 0; map_i < h2_map_dim; map_i++){
      for (int map_j = 0; map_j < h2_map_dim; map_j++){
        for (int input_map_num = 0; input_map_num < h2_maps_connected_to_h1; input_map_num++){
          int input_map_id = (k + input_map_num) % h1_maps;
          add_conv_mappings(input_map_id, input_map_num, k, map_i, map_j, h1_map_dim, h2_kernel_dim, h2_map_dim, h2_maps_connected_to_h1, h2_full_to_unique_host, h1_size, h2_unique_to_full_host, h2_map_dim * h2_map_dim, h2_null_mappings_host, h2_null_mappings_width, h2_null_set_sizes_host);
        }
      }
    }
  }



  // initalize weights and biases

  // uniform random weights between +/ # (1 / (sqrt #inputs in))^2 * 3 => uniform distribution with mean 1 and sd #inputs in ^ (-.5)
  // i rounded..
  float conv_init_bound = (1.0 / 5) * (1.0 / 5) * 3;
  float conv2_init_bound = (1.0 / 14) * (1.0 / 14) * 3;
  float h3_init_bound = (1.0 / 14) * (1.0 / 14) * 3;
  float out_init_bound = (1.0 / 6) * (1.0 / 6) * 3;
  for (int i = 0; i < W_h1_size; i++){
    // 50/50 for sign, then [0, 24/25]
    W_h1_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/conv_init_bound);
  }
  for (int i = 0; i < W_h2_size; i++){
    W_h2_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/conv2_init_bound);
  }
  for (int i = 0; i < W_h3_size; i++){
    W_h3_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/h3_init_bound);
  }
  for (int i = 0; i < W_out_size; i++){
    W_out_host[i] = (2 * (rand() % 2) - 1) * (float)rand()/(float)(RAND_MAX/out_init_bound);
  }

  for (int i = 0; i < W_h1_full_size; i++){
    W_h1_full_host[i] = 0;
  }
  for (int i = 0; i < W_h2_full_size; i++){
    W_h2_full_host[i] = 0;
  }

  // fill up the Weight matrix from unique weights
  // *** will be overwritten on first pass anyways...
  repopulate_with_unique(W_h1_full_host, W_h1_host, W_h1_size, h1_unique_to_full_host, h1_map_dim * h1_map_dim, h1_null_mappings_host, h1_null_mappings_width, input_null_ind, h1_size, input_len);
  repopulate_with_unique(W_h2_full_host, W_h2_host, W_h2_size, h2_unique_to_full_host, h2_map_dim * h2_map_dim, h2_null_mappings_host, h2_null_mappings_width, h1_null_ind, h2_size, h1_size);


  // init biases to be 0
  for (int i = 0; i < h1_size; i++){
    B_h1_host[i] = 0;
  }
  for (int i = 0; i < h2_size; i++){
    B_h2_host[i] = 0;
  }
  for (int i= 0; i < h3_size; i++){
    B_h3_host[i] = 0;
  }
  for (int i=0; i < output_len; i++){
    B_out_host[i] = 0;
  }


  // MAKE SURE MAPPINGS ARE OK...

  FILE * h1_null_mappings_file = fopen("h1_null_mappings", "w+");
  fwrite(h1_null_mappings_host, sizeof(int), (size_t) (h1_size * h1_null_mappings_width), h1_null_mappings_file);
  fclose(h1_null_mappings_file);

  FILE * h2_null_mappings_file = fopen("h2_null_mappings", "w+");
  fwrite(h2_null_mappings_host, sizeof(int), (size_t) (h2_size * h2_null_mappings_width), h2_null_mappings_file);
  fclose(h2_null_mappings_file);

  FILE * h1_null_set_sizes_file = fopen("h1_null_set_sizes", "w+");
  fwrite(h1_null_set_sizes_host, sizeof(int), (size_t) (h1_size), h1_null_set_sizes_file);
  fclose(h1_null_set_sizes_file);

  FILE * h2_null_set_sizes_file = fopen("h2_null_set_sizes", "w+");
  fwrite(h2_null_set_sizes_host, sizeof(int), (size_t) (h2_size), h2_null_set_sizes_file);
  fclose(h2_null_set_sizes_file);

  FILE * h1_unique_to_full_file = fopen("live_h1_unique_to_full", "w+");
  fwrite(h1_unique_to_full_host, sizeof(int), (size_t) (h1_maps * (h1_kernel_dim * h1_kernel_dim) * (h1_map_dim * h1_map_dim)), h1_unique_to_full_file);
  fclose(h1_unique_to_full_file);

  FILE * h2_unique_to_full_file = fopen("live_h2_unique_to_full", "w+");
  fwrite(h2_unique_to_full_host, sizeof(int), (size_t) ((h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim)), h2_unique_to_full_file);
  fclose(h2_unique_to_full_file);

  FILE * h1_full_to_unique_file = fopen("live_h1_full_to_unique", "w+");
  fwrite(h1_full_to_unique_host, sizeof(int), (size_t) (input_len * h1_size), h1_full_to_unique_file);
  fclose(h1_full_to_unique_file);

  FILE * h2_full_to_unique_file = fopen("live_h2_full_to_unique", "w+");
  fwrite(h2_full_to_unique_host, sizeof(int), (size_t) (h2_size * h1_size), h2_full_to_unique_file);
  fclose(h2_full_to_unique_file);

  FILE * w_h1_file = fopen("live_w_h1", "w+");
  fwrite(W_h1_host, sizeof(float), (size_t) (W_h1_size), w_h1_file);
  fclose(w_h1_file);

  FILE * w_h1_full_file = fopen("live_w_h1_full", "w+");
  fwrite(W_h1_full_host, sizeof(float), (size_t) (input_len * h1_size), w_h1_full_file);
  fclose(w_h1_full_file);

  FILE * w_h2_file = fopen("live_w_h2", "w+");
  fwrite(W_h2_host, sizeof(float), (size_t) (W_h2_size), w_h2_file);
  fclose(w_h2_file);

  FILE * w_h2_full_file = fopen("live_w_h2_full", "w+");
  fwrite(W_h2_full_host, sizeof(float), (size_t) (h1_size * h2_size), w_h2_full_file);
  fclose(w_h2_full_file);

  // READ FROM DATASET!!!!
  FILE * training_images_file, *training_labels_file;

  unsigned char * training_images_raw, *training_labels_raw;

  float *training_images, *training_labels;

  const char * training_images_path = "/mnt/storage/data/image_text/mnist/train-images.idx3-ubyte";
  const char * training_labels_path = "/mnt/storage/data/image_text/mnist/train-labels.idx1-ubyte";

  training_images_file = fopen(training_images_path, "rb");
  training_labels_file = fopen(training_labels_path, "rb");


  // from "http://yann.lecun.com/exdb/mnist/"
  off_t training_images_offset = 16;
  off_t training_labels_offset = 8;

  // skipping offset bytes in beginning then measuring til end = skipping offset bytes in end and measuring from start
  fseek(training_images_file, 0, SEEK_END);
  long training_images_nbytes = ftell(training_images_file);
  training_images_nbytes -= training_images_offset;
  fseek(training_images_file, training_images_offset, SEEK_SET);

  fseek(training_labels_file, 0, SEEK_END);
  long training_labels_nbytes = ftell(training_labels_file);
  training_labels_nbytes -= training_labels_offset;
  fseek(training_labels_file, training_labels_offset, SEEK_SET);

  // raw because going to downsample..
  training_images_raw = (unsigned char *) calloc(training_images_nbytes, sizeof(unsigned char));
  training_labels_raw = (unsigned char *) calloc(training_labels_nbytes, sizeof(unsigned char));

  // set to beginning...
  fseek(training_images_file, training_images_offset, SEEK_SET);
  fseek(training_labels_file, training_labels_offset, SEEK_SET);

  fread(training_images_raw, sizeof(unsigned char), (size_t) (training_images_nbytes), training_images_file);
  fclose(training_images_file);

  fread(training_labels_raw, sizeof(unsigned char), (size_t) (training_labels_nbytes), training_labels_file);
  fclose(training_labels_file);

  FILE * training_images_raw_file = fopen("training_images_raw", "wb+");
  fwrite(training_images_raw, sizeof(unsigned char), (size_t) (training_images_nbytes), training_images_raw_file);
  fclose(training_images_raw_file);

  FILE * training_labels_raw_file = fopen("training_labels_raw", "wb+");
  fwrite(training_labels_raw, sizeof(unsigned char), (size_t) (training_labels_nbytes), training_labels_raw_file);
  fclose(training_labels_raw_file);


  int training_n = 60000;
  int image_raw_dim = 28;
  int image_dim = 16;
  float ratio = float(image_raw_dim) / float(image_dim);

  int raw_floor_row, raw_floor_col, top_left, pixel_ind;
  float ave_raw_pixel, pixel_val;

  // store images as array of 16*16 images, but then additional -1 input
  training_images = (float *) calloc(training_n * input_len, sizeof(float));

  float pixel_sum;
  for (int img_num = 0; img_num < training_n; img_num++){
    pixel_sum = 0;
    //printf("Img Num: %i\n\n", img_num);
    for (int i = 0; i < image_dim; i++){
      for (int j = 0; j < image_dim; j++){
        // averaging 4 closest pixels in original image
        raw_floor_row = floor(i * ratio);
        raw_floor_col = floor(j * ratio);
        top_left = img_num * (image_raw_dim * image_raw_dim) + image_raw_dim * raw_floor_row + raw_floor_col;
        ave_raw_pixel = (float)(((float)training_images_raw[top_left] + (float)training_images_raw[top_left + 1] + (float)training_images_raw[top_left + image_raw_dim] + (float)training_images_raw[top_left + image_raw_dim + 1]) / float(4));
        // scale to be between -1 and 1
        pixel_val = ave_raw_pixel * (2 / 255.0) - 1;
        pixel_sum += pixel_val;
        // storing average pixel value into downsampled array
        pixel_ind = img_num * input_len + image_dim * i + j;
        training_images[pixel_ind] = pixel_val;
      }
    }
    // have last value in input image be a -1
    training_images[img_num * input_len + (image_dim * image_dim)] = -1;
  }
  // stored in downsampled training_images so can free now
  free(training_images_raw);

  // store image labels as series of 10 floats
  training_labels = (float *) calloc(training_n * output_len, sizeof(float));

  int label;
  for (int img_num = 0; img_num < training_n; img_num++){
    label = training_labels_raw[img_num];
    for (int dig = 0; dig < output_len; dig++){
      if (label == dig){
        training_labels[img_num * output_len + dig] = 1;
      }
      else { 
        training_labels[img_num * output_len + dig] = 0;
      }
    }
  }

  FILE * training_images_out_file = fopen("training_images", "wb+");
  fwrite(training_images, sizeof(float), (size_t) (training_n * input_len), training_images_out_file);
  fclose(training_images_out_file);

  FILE * training_labels_out_file = fopen("training_labels", "wb+");
  fwrite(training_labels, sizeof(float), (size_t) (training_n * output_len), training_labels_out_file);
  fclose(training_labels_out_file);



  // GPU variables

  // initalize a timer if want to use
  cudaEvent_t gpu_start, gpu_stop;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);


  // hidden nodes
  float *X_in, *X_h1, *X_h2, *X_h3, *X_out, *Y_out;
  float *X_in_T, *X_h1_T, *X_h2_T, *X_h3_T, *Y_out_T;
  float *sq_X_in_T, *sq_X_h1_T, *sq_X_h2_T, *sq_X_h3_T;
  float *dX_h1, *dX_h2, *dX_h3, *dX_out;
  float *sec_dX_h1, *sec_dX_h2, *sec_dX_h3, *sec_dX_out;
  float *dX_h1_activation, *dX_h2_activation, *dX_h3_activation, *dX_out_activation;
  float *sec_dX_h1_activation, *sec_dX_h2_activation, *sec_dX_h3_activation, *sec_dX_out_activation;

  float *A_h1, *A_h2, *A_h3, *A_out;
  float *softmax_jacobian;

  // weights
  float *W_h1, *W_h1_full, *W_h2, *W_h2_full, *W_h3, *W_out;
  float *W_h1_full_T, *W_h2_full_T, *W_h3_T, *W_out_T;
  float *sq_W_h1_full_T, *sq_W_h2_full_T, *sq_W_h3_T, *sq_W_out_T;
  float *dW_h1, *dW_h1_full, *dW_h2, *dW_h2_full, *dW_h3, *dW_out;
  float *sec_dW_h1, *sec_dW_h1_full, *sec_dW_h2, *sec_dW_h2_full, *sec_dW_h3, *sec_dW_out;
  float *sec_dW_h1_mem, *sec_dW_h2_mem, *sec_dW_h3_mem, *sec_dW_out_mem;
  float *W_out_step, *W_h3_step, *W_h2_step, *W_h1_step;



  // biases
  float *B_h1, *B_h2, *B_h3, *B_out;
  float *dB_h1, *dB_h2, *dB_h3, *dB_out;
  float *sec_dB_h1, *sec_dB_h2, *sec_dB_h3, *sec_dB_out;
  float *sec_dB_h1_mem, *sec_dB_h2_mem, *sec_dB_h3_mem, *sec_dB_out_mem;
  float *B_out_step, *B_h3_step, *B_h2_step, *B_h1_step;


  // weight mappings
  int *h1_null_mappings, *h2_null_mappings;
  int *h1_null_set_sizes, *h2_null_set_sizes;
  int *h1_full_to_unique, *h1_unique_to_full; 
  int *h2_full_to_unique, *h2_unique_to_full;

  // loss per sample
  float *loss;

  // predicted values
  float *predicted;

  // norms
  float *W_out_norm, *W_h3_norm, *W_h2_norm, *W_h1_norm;
  float *B_out_norm, *B_h3_norm, *B_h2_norm, *B_h1_norm;

  // ALLOCATE GPU MEMORY

  // allocate space for hidden nodes on gpu
  cudaMalloc(&X_in, input_len * batch_size * sizeof(float)); 
  cudaMalloc(&X_h1, h1_size * batch_size*sizeof(float));
  cudaMalloc(&X_h2, h2_size * batch_size*sizeof(float));
  cudaMalloc(&X_h3, h3_size * batch_size*sizeof(float));
  cudaMalloc(&X_out, output_len * batch_size*sizeof(float));
  cudaMalloc(&Y_out, output_len * batch_size* sizeof(float));

  // allocate space for node values before activation
  cudaMalloc(&A_h1, h1_size * batch_size*sizeof(float));
  cudaMalloc(&A_h2, h2_size * batch_size*sizeof(float));
  cudaMalloc(&A_h3, h3_size * batch_size*sizeof(float));
  cudaMalloc(&A_out, output_len * batch_size*sizeof(float));


  // allocate space for hidden node transposed (used in intermediate computations)
  cudaMalloc(&X_in_T, input_len * batch_size * sizeof(float)); 
  cudaMalloc(&X_h1_T, h1_size * batch_size*sizeof(float));
  cudaMalloc(&X_h2_T, h2_size * batch_size*sizeof(float));
  cudaMalloc(&X_h3_T, h3_size * batch_size*sizeof(float));
  cudaMalloc(&Y_out_T, output_len * batch_size* sizeof(float));

  // allocate space for squared transpose nodes (used to compute second deriv of weights)
  cudaMalloc(&sq_X_in_T, input_len * batch_size * sizeof(float)); 
  cudaMalloc(&sq_X_h1_T, h1_size * batch_size*sizeof(float));
  cudaMalloc(&sq_X_h2_T, h2_size * batch_size*sizeof(float));
  cudaMalloc(&sq_X_h3_T, h3_size * batch_size*sizeof(float));

  // allocate space for node gradients
  cudaMalloc(&dX_h1, h1_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h2, h2_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h3, h3_size * batch_size*sizeof(float));
  cudaMalloc(&dX_out, output_len * batch_size*sizeof(float));

  // allocate space for node gradients
  cudaMalloc(&sec_dX_h1, h1_size * batch_size*sizeof(float));
  cudaMalloc(&sec_dX_h2, h2_size * batch_size*sizeof(float));
  cudaMalloc(&sec_dX_h3, h3_size * batch_size*sizeof(float));
  cudaMalloc(&sec_dX_out, output_len * batch_size*sizeof(float));




  // allocate space for activation gradients (used in intermediate computations)
  cudaMalloc(&dX_h1_activation, h1_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h2_activation, h2_size * batch_size*sizeof(float));
  cudaMalloc(&dX_h3_activation, h3_size * batch_size*sizeof(float));
  cudaMalloc(&dX_out_activation, output_len * batch_size*sizeof(float));

  cudaMalloc(&softmax_jacobian, output_len * output_len * batch_size * sizeof(float));

  // allocate space for activation gradients (used in intermediate computations)
  cudaMalloc(&sec_dX_h1_activation, h1_size * batch_size*sizeof(float));
  cudaMalloc(&sec_dX_h2_activation, h2_size * batch_size*sizeof(float));
  cudaMalloc(&sec_dX_h3_activation, h3_size * batch_size*sizeof(float));
  cudaMalloc(&sec_dX_out_activation, output_len * batch_size*sizeof(float));


  // alllocate space for weights
  cudaMalloc(&W_h1, W_h1_size * sizeof(float));
  cudaMalloc(&W_h1_full, W_h1_full_size * sizeof(float));
  cudaMalloc(&W_h2, W_h2_size * sizeof(float));
  cudaMalloc(&W_h2_full, W_h2_full_size * sizeof(float));
  cudaMalloc(&W_h3, W_h3_size * sizeof(float));
  cudaMalloc(&W_out, W_out_size * sizeof(float));

  // alllocate space for weight_transpose (used in intermediate computations)
  cudaMalloc(&W_h1_full_T, W_h1_full_size * sizeof(float));
  cudaMalloc(&W_h2_full_T, W_h2_full_size * sizeof(float));
  cudaMalloc(&W_h3_T, W_h3_size * sizeof(float));
  cudaMalloc(&W_out_T, W_out_size * sizeof(float));

  // allocate space for squared weight_transpose (used to compute second deriv of nodes)
  cudaMalloc(&sq_W_h1_full_T, W_h1_full_size * sizeof(float));
  cudaMalloc(&sq_W_h2_full_T, W_h2_full_size * sizeof(float));
  cudaMalloc(&sq_W_h3_T, W_h3_size * sizeof(float));
  cudaMalloc(&sq_W_out_T, W_out_size * sizeof(float));

  // allocate space for weight mappings...

  cudaMalloc(&h1_null_mappings, h1_size * h1_null_mappings_width * sizeof(int));
  cudaMalloc(&h2_null_mappings, h2_size * h2_null_mappings_width * sizeof(int));

  cudaMalloc(&h1_null_set_sizes, h1_size * sizeof(int));
  cudaMalloc(&h2_null_set_sizes, h2_size * sizeof(int));

  cudaMalloc(&h1_full_to_unique, input_len * h1_size * sizeof(int));
  cudaMalloc(&h1_unique_to_full, (h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim) * sizeof(int));

  cudaMalloc(&h2_full_to_unique, h1_size * h2_size * sizeof(int));
  cudaMalloc(&h2_unique_to_full, (h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim) * sizeof(int));


  // allocate space for weight gradients
  cudaMalloc(&dW_h1, W_h1_size * sizeof(float));
  cudaMalloc(&dW_h1_full, W_h1_full_size * sizeof(float));
  cudaMalloc(&dW_h2, W_h2_size * sizeof(float));
  cudaMalloc(&dW_h2_full, W_h2_full_size * sizeof(float));
  cudaMalloc(&dW_h3, W_h3_size * sizeof(float));
  cudaMalloc(&dW_out, W_out_size * sizeof(float));

  // second deriv of weights
  cudaMalloc(&sec_dW_h1, W_h1_size * sizeof(float));
  cudaMalloc(&sec_dW_h1_full, W_h1_full_size * sizeof(float));
  cudaMalloc(&sec_dW_h2, W_h2_size * sizeof(float));
  cudaMalloc(&sec_dW_h2_full, W_h2_full_size * sizeof(float));
  cudaMalloc(&sec_dW_h3, W_h3_size * sizeof(float));
  cudaMalloc(&sec_dW_out, W_out_size * sizeof(float));

  cudaMalloc(&sec_dW_h1_mem, W_h1_size * sizeof(float));
  cudaMalloc(&sec_dW_h2_mem, W_h2_size * sizeof(float));
  cudaMalloc(&sec_dW_h3_mem, W_h3_size * sizeof(float));
  cudaMalloc(&sec_dW_out_mem, W_out_size * sizeof(float));


  // allocate space for biases
  cudaMalloc(&B_h1, h1_size * sizeof(float));
  cudaMalloc(&B_h2, h2_size * sizeof(float));
  cudaMalloc(&B_h3, h3_size * sizeof(float));
  cudaMalloc(&B_out, output_len * sizeof(float));

  // allocate space for bias gradients
  cudaMalloc(&dB_h1, h1_size * sizeof(float));
  cudaMalloc(&dB_h2, h2_size * sizeof(float));
  cudaMalloc(&dB_h3, h3_size * sizeof(float));
  cudaMalloc(&dB_out, output_len * sizeof(float));

  // allocate space for bias gradients
  cudaMalloc(&sec_dB_h1, h1_size * sizeof(float));
  cudaMalloc(&sec_dB_h2, h2_size * sizeof(float));
  cudaMalloc(&sec_dB_h3, h3_size * sizeof(float));
  cudaMalloc(&sec_dB_out, output_len * sizeof(float));

  cudaMalloc(&sec_dB_h1_mem, h1_size * sizeof(float));
  cudaMalloc(&sec_dB_h2_mem, h2_size * sizeof(float));
  cudaMalloc(&sec_dB_h3_mem, h3_size * sizeof(float));
  cudaMalloc(&sec_dB_out_mem, output_len * sizeof(float));


    // step sizes for each variable..
  cudaMalloc(&W_out_step, W_out_size * sizeof(float));
  cudaMalloc(&W_h3_step, W_h3_size * sizeof(float));
  cudaMalloc(&W_h2_step, W_h2_size * sizeof(float));
  cudaMalloc(&W_h1_step, W_h1_size * sizeof(float));

  cudaMalloc(&B_h1_step, h1_size * sizeof(float));
  cudaMalloc(&B_h2_step, h2_size * sizeof(float));
  cudaMalloc(&B_h3_step, h3_size * sizeof(float));
  cudaMalloc(&B_out_step, output_len * sizeof(float));


  // allocate space to store values for loss function per sample
  cudaMalloc(&loss, batch_size * sizeof(float));

  cudaMalloc(&predicted, batch_size * sizeof(float));


  // norms

  cudaMalloc(&W_out_norm, sizeof(float));
  cudaMalloc(&W_h3_norm, sizeof(float));
  cudaMalloc(&W_h2_norm, sizeof(float));
  cudaMalloc(&W_h1_norm, sizeof(float));

  cudaMalloc(&B_out_norm, sizeof(float));
  cudaMalloc(&B_h3_norm, sizeof(float));
  cudaMalloc(&B_h2_norm, sizeof(float));
  cudaMalloc(&B_h1_norm, sizeof(float));


  // COPY VALUES FROM CPU

  // initalized weights and biases
  cudaMemcpy(W_h1, W_h1_host, W_h1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_h2, W_h2_host, W_h2_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_h3, W_h3_host, W_h3_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_out, W_out_host, W_out_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(W_h1_full, W_h1_full_host, W_h1_full_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(W_h2_full, W_h2_full_host, W_h2_full_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(B_h1, B_h1_host, h1_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_h2, B_h2_host, h2_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_h3, B_h3_host, h3_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_out, B_out_host, output_len * sizeof(float), cudaMemcpyHostToDevice);

  // conv weight mappings
  cudaMemcpy(h1_null_mappings, h1_null_mappings_host, h1_size * h1_null_mappings_width * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h2_null_mappings, h2_null_mappings_host, h2_size * h2_null_mappings_width * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(h1_null_set_sizes, h1_null_set_sizes_host, h1_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h2_null_set_sizes, h2_null_set_sizes_host, h2_size * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(h1_full_to_unique, h1_full_to_unique_host, input_len * h1_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h1_unique_to_full, h1_unique_to_full_host, (h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(h2_full_to_unique, h2_full_to_unique_host, h1_size * h2_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h2_unique_to_full, h2_unique_to_full_host, (h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim) * sizeof(int), cudaMemcpyHostToDevice);

  cudaEventRecord(gpu_start);

  // TRAINNNNNN

  float h1_from_unique_size = (h1_maps * (h1_kernel_dim * h1_kernel_dim)) * (h1_map_dim * h1_map_dim);
  float h1_from_unique_width = (h1_map_dim * h1_map_dim);
  float h2_from_unique_size = (h2_maps * h2_maps_connected_to_h1 * (h2_kernel_dim * h2_kernel_dim)) * (h2_map_dim * h2_map_dim);
  float h2_from_unique_width = (h2_map_dim * h2_map_dim);

  float * A_h1_host = (float *)malloc(h1_size * batch_size * sizeof(float));
  float * A_h2_host = (float *)malloc(h2_size * batch_size * sizeof(float));
  float * A_h3_host = (float *)malloc(h3_size * batch_size * sizeof(float));
  float * A_out_host = (float *)malloc(output_len * batch_size * sizeof(float));

  float *W_out_norm_host = (float *)malloc(sizeof(float));
  float *W_h3_norm_host = (float *)malloc(sizeof(float));
  float *W_h2_norm_host = (float *)malloc(sizeof(float));
  float *W_h1_norm_host = (float *)malloc(sizeof(float));

  float *B_out_norm_host = (float *)malloc(sizeof(float));
  float *B_h3_norm_host = (float *)malloc(sizeof(float));
  float *B_h2_norm_host = (float *)malloc(sizeof(float));
  float *B_h1_norm_host = (float *)malloc(sizeof(float));


  float *sec_dW_out_host = (float *)malloc(W_out_size * sizeof(float));
  float *sec_dW_h3_host = (float *)malloc(W_h3_size * sizeof(float));
  float *sec_dW_h2_host = (float *)malloc(W_h2_size * sizeof(float));
  float *sec_dW_h1_host = (float *)malloc(W_h1_size * sizeof(float));

  float *sec_dB_out_host = (float *)malloc(output_len * sizeof(float));
  float *sec_dB_h3_host = (float *)malloc(h3_size * sizeof(float));
  float *sec_dB_h2_host = (float *)malloc(h2_size * sizeof(float));
  float *sec_dB_h1_host = (float *)malloc(h1_size * sizeof(float));


  float *W_out_step_host = (float *)malloc(W_out_size * sizeof(float));
  float *W_h3_step_host = (float *)malloc(W_h3_size * sizeof(float));
  float *W_h2_step_host = (float *)malloc(W_h2_size * sizeof(float));
  float *W_h1_step_host = (float *)malloc(W_h1_size * sizeof(float));

  float *B_out_step_host = (float *)malloc(output_len * sizeof(float));
  float *B_h3_step_host = (float *)malloc(h3_size * sizeof(float));
  float *B_h2_step_host = (float *)malloc(h2_size * sizeof(float));
  float *B_h1_step_host = (float *)malloc(h1_size * sizeof(float));

  float *softmax_jacobian_host = (float *)malloc(output_len * output_len * batch_size * sizeof(float));

  float *sec_dX_out_activation_host = (float *)malloc(output_len * batch_size *sizeof(float));
  float *sec_dX_h3_activation_host = (float *)malloc(h3_size * batch_size *sizeof(float));
  float *sec_dX_h2_activation_host = (float *)malloc(h2_size * batch_size *sizeof(float));
  float *sec_dX_h1_activation_host = (float *)malloc(h1_size * batch_size *sizeof(float));




  int n_batches = ceil((float)training_n / batch_size);
  for (int cnt = 0; cnt < repeat_n; cnt++){
    float totalLoss = 0;
    float n_wrong = 0;
    printf("\nDataset Iteration: %i\n\n", cnt);
    for (int batch_i = 0; batch_i < n_batches; batch_i++){

        if (batch_i % 1000 == 0){
          printf("Batch #: %d\n", batch_i);
        }

        int rand_batch_start_ind = rand() % (training_n - batch_size);

        rand_batch_start_ind = batch_i;


        // get new batch
        memcpy(X_in_host, training_images + rand_batch_start_ind * input_len, batch_size * input_len * sizeof(float));
        memcpy(Y_out_host, training_labels + rand_batch_start_ind * output_len, batch_size * output_len * sizeof(float));
        // read in as consective images (so pixels are rows). want to transpose, then send back to host
        cudaMemcpy(X_in_T, X_in_host, input_len *batch_size* sizeof(float), cudaMemcpyHostToDevice);
        transposeSimp<<< SM_COUNT, ceil((float)input_len * batch_size / SM_COUNT)>>> (input_len * batch_size, input_len, X_in_T, X_in);
        
        cudaMemcpy(X_in_host, X_in, input_len *batch_size* sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(input_len * batch_size, X_in_host, "X_in");
        
        // read in as consective sequences of output lables, want to transpose
        cudaMemcpy(Y_out_T, Y_out_host, output_len*batch_size*sizeof(float), cudaMemcpyHostToDevice);
        transposeSimp<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>> (output_len * batch_size, output_len, Y_out_T, Y_out);
        
        cudaMemcpy(Y_out_host, Y_out, output_len *batch_size* sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len * batch_size, Y_out_host, "Y_out");

        
        // FORWARD PASS

        // COMPUTE FULL MATRICES FROM UNIQUE WEIGHTS 
        // conv layer weights W_h1_full, W_h2_full

        // first fill in the 1-1 mappings for connections with non-null input
        repopulate_with_unique_non_null_gpu<<<SM_COUNT, ceil((float)(h1_from_unique_size) / SM_COUNT)>>>(h1_from_unique_size, W_h1_full, W_h1, W_h1_size, h1_unique_to_full, h1_from_unique_width);
        repopulate_with_unique_non_null_gpu<<<SM_COUNT, ceil((float)(h2_from_unique_size) / SM_COUNT)>>>(h2_from_unique_size, W_h2_full, W_h2, W_h2_size, h2_unique_to_full, h2_from_unique_width);

        // now create virtual weights which represent aggregation of weight values for all weight inds going from null -> output node (for all output nodes)
        repopulate_with_unique_null_gpu<<<SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, W_h1_full, W_h1, W_h1_size, h1_null_mappings, h1_null_mappings_width, input_null_ind, input_len);
        repopulate_with_unique_null_gpu<<<SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, W_h2_full, W_h2, W_h2_size, h2_null_mappings, h2_null_mappings_width, h1_null_ind, h1_size);

        /// MOST MEMCPYS ARE FOR DEBUGGING ON CPU PURPOSES (note "DeviceToHost")
        cudaMemcpy(W_h1_host, W_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);

        //my_write_file(W_h1_size, W_h1_host, "W_h1");

        cudaMemcpy(W_h1_full_host, W_h1_full, W_h1_full_size * sizeof(float), cudaMemcpyDeviceToHost);

        //my_write_file(W_h1_full_size, W_h1_full_host, "W_h1_full");

        matMulSimp<<< 4 * SM_COUNT, ceil((float)h1_size * batch_size / (4 * SM_COUNT)) >>>(h1_size, input_len, batch_size, W_h1_full, X_in, X_h1);
        addBias <<< 4 * SM_COUNT, ceil((float)h1_size * batch_size / (4 * SM_COUNT)) >>>(h1_size * batch_size, batch_size, X_h1, B_h1);

        cudaMemcpy(B_h1_host, B_h1, h1_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        //my_write_file(h1_size, B_h1_host, "B_h1");

        // set rowValue here for A
        setRowVal<<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, h1_size - 1, X_h1, -1.0);
        cudaMemcpy(A_h1, X_h1, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(A_h1_host, A_h1, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h1_size * batch_size, A_h1_host, "A_h1");

        activate<<< 4 * SM_COUNT, ceil((float)h1_size * batch_size / (4 * SM_COUNT))>>>(h1_size * batch_size, batch_size, X_h1);
        
        // set constant of -1 to last val in h1 for all samples in batch
        setRowVal<<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, h1_size - 1, X_h1, -1.0);

        cudaMemcpy(X_h1_host, X_h1, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h1_size * batch_size, X_h1_host, "X_h1");

          // h1 to h2

        cudaMemcpy(W_h2_host, W_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_h2_size, W_h2_host, "W_h2");


        cudaMemcpy(W_h2_full_host, W_h2_full, W_h2_full_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_h2_full_size, W_h2_full_host, "W_h2_full");

        matMulSimp<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT) >>>(h2_size, h1_size, batch_size, W_h2_full, X_h1, X_h2);
        addBias<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT) >>>(h2_size * batch_size, batch_size, X_h2, B_h2);
        
        cudaMemcpy(B_h2_host, B_h2, h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h2_size, B_h2_host, "B_h2");

        cudaMemcpy(A_h2, X_h2, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(A_h2_host, A_h2, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h2_size * batch_size, A_h2_host, "A_h2");

        activate<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size * batch_size, batch_size, X_h2);
        
        cudaMemcpy(X_h2_host, X_h2, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h2_size * batch_size, X_h2_host, "X_h2");


          // h2 to h3

        cudaMemcpy(W_h3_host, W_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_h3_size, W_h3_host, "W_h3");

        matMulSimp<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT) >>>(h3_size, h2_size, batch_size, W_h3, X_h2, X_h3);
        addBias <<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT) >>>(h3_size * batch_size, batch_size, X_h3, B_h3);
        
        cudaMemcpy(B_h3_host, B_h3, h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h3_size, B_h3_host, "B_h3");


        cudaMemcpy(A_h3, X_h3, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(A_h3_host, A_h3, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h3_size * batch_size, A_h3_host, "A_h3");

        activate<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size * batch_size, batch_size, X_h3);

        cudaMemcpy(X_h3_host, X_h3, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h3_size * batch_size, X_h3_host, "X_h3");

          // h3 to output

        cudaMemcpy(W_out_host, W_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_out_size, W_out_host, "W_out");

        matMulSimp<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len, h3_size, batch_size, W_out, X_h3, X_out);
        addBias <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>> (output_len * batch_size, batch_size, X_out, B_out);

        cudaMemcpy(B_out_host, B_out, output_len * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len, B_out_host, "B_out");

        cudaMemcpy(A_out, X_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(A_out_host, A_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len * batch_size, A_out_host, "A_out");
        
        //activate <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len * batch_size, batch_size, X_out);
        
        softMax <<< SM_COUNT, ceil((float)batch_size / SM_COUNT) >>> (batch_size, output_len, X_out);
        
        cudaMemcpy(X_out_host, X_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len * batch_size, X_out_host, "X_out");
        

        computeLoss <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, Y_out, loss);
        cudaMemcpy(loss_host, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch_size; i++){
          totalLoss += loss_host[i];
        }

        makePredict <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, predicted);

        cudaMemcpy(predicted_host, predicted, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch_size; i++){
          if (predicted_host[i] != training_labels_raw[rand_batch_start_ind + i]){
            n_wrong++;
          }
        }  
        
        // BACK PROP => want dW_out, dW_h3, dW_h2, dW_h1 + dB_out, dB_h3, dB_h2, dB_h1

        
        // compute dX_out (SOFTMAX LOSS!)
        matSub<<<SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len*batch_size, X_out, Y_out, dX_out);
        matScale <<< SM_COUNT, ceil((float)output_len * batch_size/ SM_COUNT)>>>(output_len * batch_size, dX_out, 1.0 / (batch_size * output_len));
        //matScale <<< SM_COUNT, ceil((float)output_len * batch_size/ SM_COUNT)>>>(output_len * batch_size, dX_out, 1.0 / (output_len));
        
        cudaMemcpy(dX_out_host, dX_out, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len * batch_size, dX_out_host, "dX_out");
     
        // set second derivative of output layer = 1 (i.e. sec_dX_out = all 1's)
        
        setVal <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len*batch_size, sec_dX_out, 1.0 / (batch_size * output_len));
        //setVal <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len*batch_size, sec_dX_out, 1.0 / (output_len));

    
        // prep for next layer back ...

        // compute softmax jacobian
        softMaxJacobian <<< SM_COUNT, ceil((float)output_len * output_len * batch_size / SM_COUNT)>>> (output_len * output_len * batch_size, batch_size, output_len, X_out, softmax_jacobian);

        cudaMemcpy(softmax_jacobian_host, softmax_jacobian, output_len * output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        // my_write_file(output_len * output_len * batch_size, softmax_jacobian_host, "softmax_jacobian");

        activationDerivSoft<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len * batch_size, batch_size, output_len, dX_out, softmax_jacobian, dX_out_activation);
        activationSecondDerivSoft <<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len * batch_size, sec_dX_out, X_out, sec_dX_out_activation);

        // activationDeriv<<< SM_COUNT, ceil((float)output_len * batch_size/ SM_COUNT)>>>(output_len * batch_size, dX_out, A_out, dX_out_activation);
        // activationSecondDeriv<<< SM_COUNT, ceil((float)output_len * batch_size / SM_COUNT)>>>(output_len * batch_size, sec_dX_out, A_out, sec_dX_out_activation);

        cudaMemcpy(dX_out_activation_host, dX_out_activation, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len * batch_size, dX_out_activation_host, "dX_out_activation");


        cudaMemcpy(sec_dX_out_activation_host, sec_dX_out_activation, output_len * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(output_len * batch_size, sec_dX_out_activation_host, "sec_dX_out_activation");


        

        // biases dB_out and sec_dB_out
        biasDerivs <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>> (output_len, batch_size, dX_out_activation, dB_out);
        biasDerivs <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>> (output_len, batch_size, sec_dX_out_activation, sec_dB_out);

       
        // compute dW_out

        // get X_h3_T
        transposeSimp<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size * batch_size, batch_size, X_h3, X_h3_T);

        cudaMemcpy(X_h3_T_host, X_h3_T, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // VERY STUPID GRID/BLOCK DIMENSIONS (should fix when the matrix multiply kernel switches...)
        matMulSimp<<< SM_COUNT, ceil((float)output_len * h3_size / SM_COUNT) >>>(output_len, batch_size, h3_size, dX_out_activation, X_h3_T, dW_out);
        // take average of across all samples in batch
        //matScale <<< SM_COUNT, ceil((float)W_out_size/ SM_COUNT)>>>(W_out_size, dW_out, 1.0 / batch_size);
        
        // compute sec_dW_out
        elementSquare<<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size * batch_size, X_h3_T, sq_X_h3_T);
        matMulSimp<<< SM_COUNT, ceil((float)output_len * h3_size / SM_COUNT) >>>(output_len, batch_size, h3_size, sec_dX_out_activation, sq_X_h3_T, sec_dW_out);
        // take average of across all samples in batch
        //matScale <<< SM_COUNT, ceil((float)W_out_size/ SM_COUNT)>>>(W_out_size, sec_dW_out, 1.0 / batch_size);
        
        cudaMemcpy(dW_out_host, dW_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_out_size, dW_out_host, "dW_out");

        
        // compute dX_h3
        transposeSimp<<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, h3_size, W_out, W_out_T);
        matMulSimp<<<SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size, output_len, batch_size, W_out_T, dX_out_activation, dX_h3);
        
        cudaMemcpy(dX_h3_host, dX_h3, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h3_size * batch_size, dX_h3_host, "dX_h3");

        // compute sec_dX_h3
        elementSquare<<<SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, W_out_T, sq_W_out_T);
        matMulSimp<<<SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size, output_len, batch_size, sq_W_out_T, sec_dX_out_activation, sec_dX_h3);


        // prep for next layer back...
        activationDeriv<<< SM_COUNT, ceil((float)h3_size * batch_size/ SM_COUNT)>>>(h3_size * batch_size, dX_h3, A_h3, dX_h3_activation);
        activationSecondDeriv <<< SM_COUNT, ceil((float)h3_size * batch_size / SM_COUNT)>>>(h3_size * batch_size, sec_dX_h3, A_h3, sec_dX_h3_activation);

        cudaMemcpy(dX_h3_activation_host, dX_h3_activation, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h3_size * batch_size, dX_h3_activation_host, "dX_h3_activation");

        cudaMemcpy(sec_dX_h3_activation_host, sec_dX_h3_activation, h3_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h3_size * batch_size, sec_dX_h3_activation_host, "sec_dX_h3_activation");

        // biases dB_h3 and sec_dB_h3
        biasDerivs <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>> (h3_size, batch_size, dX_h3_activation, dB_h3);
        biasDerivs <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>> (h3_size, batch_size, sec_dX_h3_activation, sec_dB_h3);
        
        

        // compute dW_h3
        transposeSimp<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size * batch_size, batch_size, X_h2, X_h2_T);
        matMulSimp<<< SM_COUNT, ceil((float)h3_size * h2_size / SM_COUNT) >>>(h3_size, batch_size, h2_size, dX_h3_activation, X_h2_T, dW_h3);
        // take average of across all samples in batch
        //matScale <<< SM_COUNT, ceil((float)W_h3_size/ SM_COUNT)>>>(W_h3_size, dW_h3, 1.0 / batch_size);

        // compute sec_dW_h3
        elementSquare<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size * batch_size, X_h2_T, sq_X_h2_T);
        matMulSimp<<< SM_COUNT, ceil((float)h3_size * h2_size / SM_COUNT) >>>(h3_size, batch_size, h2_size, sec_dX_h3_activation, sq_X_h2_T, sec_dW_h3);
        // take average of across all samples in batch
        //matScale <<< SM_COUNT, ceil((float)W_h3_size/ SM_COUNT)>>>(W_h3_size, sec_dW_h3, 1.0 / batch_size);
        
        cudaMemcpy(dW_h3_host, dW_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_h3_size, dW_h3_host, "dW_h3");

        // compute dX_h2
        transposeSimp<<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, h2_size, W_h3, W_h3_T);
        matMulSimp<<<SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size, h3_size, batch_size, W_h3_T, dX_h3_activation, dX_h2);
        
        cudaMemcpy(dX_h2_host, dX_h2, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h2_size * batch_size, dX_h2_host, "dX_h2");

        // compute sec_dX_h2
        elementSquare<<<SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, W_h3_T, sq_W_h3_T);
        matMulSimp<<<SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size, h3_size, batch_size, sq_W_h3_T, sec_dX_h3_activation, sec_dX_h2);
        
                
        // prep for next layer back...
        activationDeriv<<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size *batch_size, dX_h2, A_h2, dX_h2_activation);
        activationSecondDeriv <<< SM_COUNT, ceil((float)h2_size * batch_size / SM_COUNT)>>>(h2_size * batch_size, sec_dX_h2, A_h2, sec_dX_h2_activation);

        cudaMemcpy(dX_h2_activation_host, dX_h2_activation, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h2_size * batch_size, dX_h2_activation_host, "dX_h2_activation");

        cudaMemcpy(sec_dX_h2_activation_host, sec_dX_h2_activation, h2_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h2_size * batch_size, sec_dX_h2_activation_host, "sec_dX_h2_activation");

        // biases dB_h2 and sec_dB_h2
        biasDerivs <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>> (h2_size, batch_size, dX_h2_activation, dB_h2);
        biasDerivs <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>> (h2_size, batch_size, sec_dX_h2_activation, sec_dB_h2);


        // NOW AT CONVOLUTION LAYER, so need to aggregate shared weights...

        // first compute derivs of full matrix, then condense to how unique vectors change. then add back to full matrix

        // compute dW_h2
        transposeSimp<<< 4 * SM_COUNT, ceil((float)h1_size * batch_size / (4 * SM_COUNT))>>>(h1_size * batch_size, batch_size, X_h1, X_h1_T);
        matMulSimp<<< 2 * SM_COUNT, ceil((float)h2_size * h1_size / (2 * SM_COUNT)) >>>(h2_size, batch_size, h1_size, dX_h2_activation, X_h1_T, dW_h2_full);
        // take average of across all samples in batch
        //matScale <<< 2 * SM_COUNT, ceil((float)W_h2_full_size/ (2 * SM_COUNT))>>>(W_h2_full_size, dW_h2_full, 1.0 / batch_size);

        // Convert the Full Matrix derivative to unique weight changes in 2 steps (normal vs. connections coming from null input)

        // first sum over weights coming from non-null input, which have 1-1 mapping with ind->value in matrix
        fullWeightDerivToUniqueNonNull<<< 2 * SM_COUNT, ceil((float)W_h2_full_size / (2 * SM_COUNT))>>>(W_h2_full_size, h2_full_to_unique, dW_h2_full, dW_h2);
        // now go over the set of weights which map from null input to output node. distribute this virtual null weight deriv among this set 
        fullWeightDerivToUniqueNull<<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, h1_size, dW_h2_full, dW_h2, h2_null_mappings, h2_null_mappings_width, h2_null_set_sizes, h1_null_ind);

        // take average deriv by dividing by # of connections using shared weight (kernel_2 dim ^2)
        matScale <<< SM_COUNT, ceil((float)W_h2_size/ SM_COUNT)>>>(W_h2_size, dW_h2, 1.0 / (h2_kernel_dim * h2_kernel_dim * h2_maps_connected_to_h1));
        
        cudaMemcpy(dW_h2_host, dW_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_h2_size, dW_h2_host, "dW_h2");


        // compute sec_dW_h2
        elementSquare<<< 4 * SM_COUNT, ceil((float)h1_size * batch_size / (4 * SM_COUNT))>>>(h1_size * batch_size, X_h1_T, sq_X_h1_T);
        matMulSimp<<< 2 * SM_COUNT, ceil((float)h2_size * h1_size / (2 * SM_COUNT)) >>>(h2_size, batch_size, h1_size, sec_dX_h2_activation, sq_X_h1_T, sec_dW_h2_full);
        // take average of across all samples in batch
        //matScale <<<  2 * SM_COUNT, ceil((float)W_h2_full_size/ (2 * SM_COUNT))>>>(W_h2_full_size, sec_dW_h2_full, 1.0 / batch_size);

        // first sum over weights coming from non-null input, which have 1-1 mapping with ind->value in matrix
        fullWeightDerivToUniqueNonNull<<< 2 * SM_COUNT, ceil((float)W_h2_full_size / (2 * SM_COUNT))>>>(W_h2_full_size, h2_full_to_unique, sec_dW_h2_full, sec_dW_h2);
        // now go over the set of weights which map from null input to output node. distribute this virtual null weight deriv among this set 
        fullWeightDerivToUniqueNull<<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, h1_size, sec_dW_h2_full, sec_dW_h2, h2_null_mappings, h2_null_mappings_width, h2_null_set_sizes, h1_null_ind);
        
        // take average deriv by dividing by # of connections using shared weight (kernel_2 dim ^2)
        matScale <<< SM_COUNT, ceil((float)W_h2_size/ SM_COUNT)>>>(W_h2_size, sec_dW_h2, 1.0 / (h2_kernel_dim * h2_kernel_dim * h2_maps_connected_to_h1));

        
        

        transposeSimp<<< 2 * SM_COUNT, ceil((float)W_h2_full_size / (2 * SM_COUNT)) >>>(W_h2_full_size, h1_size, W_h2_full, W_h2_full_T);
        matMulSimp<<<2 * SM_COUNT, ceil((float)h1_size * batch_size / (2 * SM_COUNT))>>>(h1_size, h2_size, batch_size, W_h2_full_T, dX_h2_activation, dX_h1);
        
        cudaMemcpy(dX_h1_host, dX_h1, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h1_size * batch_size, dX_h1_host, "dX_h1");

        // compute sec_dX_h1
        elementSquare<<<2 * SM_COUNT, ceil((float)W_h2_full_size / (2 * SM_COUNT))>>>(W_h2_full_size, W_h2_full_T, sq_W_h2_full_T);
        matMulSimp<<<2 * SM_COUNT, ceil((float)h1_size * batch_size / (2 * SM_COUNT))>>>(h1_size, h2_size, batch_size, sq_W_h2_full_T, sec_dX_h2_activation, sec_dX_h1);

        // prep for next layer back...
        activationDeriv<<< 2 * SM_COUNT, ceil((float)h1_size * batch_size / (2 * SM_COUNT))>>>(h1_size *batch_size, dX_h1, A_h1, dX_h1_activation);
        activationSecondDeriv <<< 2 * SM_COUNT, ceil((float)h1_size * batch_size / (2 * SM_COUNT))>>>(h1_size * batch_size, sec_dX_h1, A_h1, sec_dX_h1_activation);

        cudaMemcpy(dX_h1_activation_host, dX_h1_activation, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h1_size * batch_size, dX_h1_activation_host, "dX_h1_activation");

        cudaMemcpy(sec_dX_h1_activation_host, sec_dX_h1_activation, h1_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(h1_size * batch_size, sec_dX_h1_activation_host, "sec_dX_h1_activation");

        // bises dB_h1 and sec_dB_h1
        biasDerivs <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>> (h1_size, batch_size, dX_h1_activation, dB_h1);
        biasDerivs <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>> (h1_size, batch_size, sec_dX_h1_activation, sec_dB_h1);


        // ANOTHER CONVOLUATION LAYER BETWEEN H1 and H2

        // compute dW_h1
        transposeSimp<<< 4 * SM_COUNT, ceil((float)input_len * batch_size / (4 * SM_COUNT))>>>(input_len * batch_size, batch_size, X_in, X_in_T);
        matMulSimp<<< 4 * SM_COUNT, ceil((float)h1_size * input_len / (4 * SM_COUNT)) >>>(h1_size, batch_size, input_len, dX_h1_activation, X_in_T, dW_h1_full);
        // take average of across all samples in batch
        //matScale <<< 4 * SM_COUNT, ceil((float)W_h1_full_size/ (4 * SM_COUNT))>>>(W_h1_full_size, dW_h1_full, 1.0 / batch_size);

        // Convert the Full Matrix derivative to unique weight changes in 2 steps (normal vs. connections coming from null input)

        // first sum over weights coming from non-null input, which have 1-1 mapping with ind->value in matrix
        fullWeightDerivToUniqueNonNull<<< 4 * SM_COUNT, ceil((float)W_h1_full_size / (4 * SM_COUNT))>>>(W_h1_full_size, h1_full_to_unique, dW_h1_full, dW_h1);
        // now go over the set of weights which map from null input to output node. distribute this virtual null weight deriv among this set 
        fullWeightDerivToUniqueNull<<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, input_len, dW_h1_full, dW_h1, h1_null_mappings, h1_null_mappings_width, h1_null_set_sizes, input_null_ind);

        // take average deriv
        matScale <<< SM_COUNT, ceil((float)W_h1_size/ SM_COUNT)>>>(W_h1_size, dW_h1, 1.0 / (h1_kernel_dim * h1_kernel_dim));

        cudaMemcpy(dW_h1_host, dW_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);
        //my_write_file(W_h1_size, dW_h1_host, "dW_h1");

        // compute sec_dW_h1
        elementSquare<<< 4 * SM_COUNT, ceil((float)input_len * batch_size / (4 * SM_COUNT))>>>(input_len * batch_size, X_in_T, sq_X_in_T);
        matMulSimp<<< 4 * SM_COUNT, ceil((float)input_len * h1_size / (4 * SM_COUNT)) >>>(h1_size, batch_size, input_len, sec_dX_h1_activation, sq_X_in_T, sec_dW_h1_full);
        // take average of across all samples in batch
        //matScale <<< 4 * SM_COUNT, ceil((float)W_h1_full_size/ (4 * SM_COUNT))>>>(W_h1_full_size, sec_dW_h1_full, 1.0 / batch_size);
       
        // first sum over weights coming from non-null input, which have 1-1 mapping with ind->value in matrix
        fullWeightDerivToUniqueNonNull<<< 4 * SM_COUNT, ceil((float)W_h1_full_size / (4 * SM_COUNT))>>>(W_h1_full_size, h1_full_to_unique, sec_dW_h1_full, sec_dW_h1);
        // now go over the set of weights which map from null input to output node. distribute this virtual null weight deriv among this set 
        fullWeightDerivToUniqueNull<<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, input_len, sec_dW_h1_full, sec_dW_h1, h1_null_mappings, h1_null_mappings_width, h1_null_set_sizes, input_null_ind);

        // take average secon deriv by dividing by # of connections using shared weight (kernel_2 dim ^2)
        matScale <<< SM_COUNT, ceil((float)W_h1_size/ SM_COUNT)>>>(W_h1_size, sec_dW_h1, 1.0 / (h1_kernel_dim * h1_kernel_dim));

        

        /// DEBUG STUFF....

        if ((batch_i % 1000 == 0)) {
          
          printf("BATCH #: %i\n", batch_i);
          // printf("\n\nX_in MATRIX:\n\n");
          // for (int i = 0; i < input_len * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", X_in_host[i]);
          // }

          // printf("\n\nW_h1 values:\n\n");
          // for (int i = 0; i < W_h1_size; i++){
          //   printf("%f\n", W_h1_host[i]);
          // }

          // printf("\n\nB_h1 MATRIX:\n\n");
          // for (int i = 0; i < h1_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", B_h1_host[i]);
          // }


          // printf("\n\nX_h1 MATRIX:\n\n");
          // for (int i = 0; i < h1_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", X_h1_host[i]);
          // }

          // printf("\n\nW_h2 values:\n\n");
          // for (int i = 0; i < W_h2_size; i++){
          //   printf("%f\n", W_h2_host[i]);
          // }

          // printf("\n\nB_h2 MATRIX:\n\n");
          // for (int i = 0; i < h2_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", B_h2_host[i]);
          // }

          // printf("\n\nX_h2 MATRIX:\n\n");
          // for (int i = 0; i < h2_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", X_h2_host[i]);
          // }

          // printf("\n\nW_h3 MATRIX:\n\n");
          // for (int i = 0; i < W_h3_size; i++){
          //   if ((i % h2_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", W_h3_host[i]);
          // }

          // printf("\n\nB_h3 MATRIX:\n\n");
          // for (int i = 0; i < h3_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", B_h3_host[i]);
          // }
          
          // printf("\n\nX_h3 MATRIX:\n\n");
          // for (int i = 0; i < h3_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", X_h3_host[i]);
          // }



          printf("\n\nW_out MATRIX:\n\n");
          for (int i = 0; i < W_out_size; i++){
            if ((i % h3_size) == 0) {
              printf("\n");
            }
            printf("%f ", W_out_host[i]);
          }

          printf("\n\nB_out MATRIX:\n\n");
          for (int i = 0; i < output_len; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", B_out_host[i]);
          }


          printf("\n\nA_out MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", A_out_host[i]);
          }

          printf("\n\nX OUT MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n\n");
            }
            printf("%f ", X_out_host[i]);
          }

          printf("\n\nY OUT MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", Y_out_host[i]);
          }

          printf("\n\ndX OUT MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", dX_out_host[i]);
          }

          printf("\n\ndX OUT ACTIVATION MATRIX:\n\n");
          for (int i = 0; i < output_len * batch_size; i++){
            if ((i % batch_size) == 0) {
              printf("\n");
            }
            printf("%f ", dX_out_activation_host[i]);
          }

          printf("\n\nGRADIENT LAST WEIGHT MATRIX:\n\n");
          for (int i = 0; i < W_out_size; i++){
            if ((i % h3_size) == 0) {
              printf("\n");
            }
            printf("%f ", dW_out_host[i]);
          }

          // printf("\n\ndX H3 MATRIX:\n\n");
          // for (int i = 0; i < h3_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", dX_h3_host[i]);
          // }

          // printf("\n\nA_h3 MATRIX:\n\n");
          // for (int i = 0; i < h3_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", A_h3_host[i]);
          // }

          // printf("\n\ndX h3 ACTIVATION MATRIX:\n\n");
          // for (int i = 0; i < h3_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", dX_h3_activation_host[i]);
          // }



          // printf("\n\ndX h2 ACTIVATION MATRIX:\n\n");
          // for (int i = 0; i < h2_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", dX_h2_activation_host[i]);
          // }


          // printf("\n\ndX h1 ACTIVATION MATRIX:\n\n");
          // for (int i = 0; i < h1_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", dX_h1_activation_host[i]);
          // }

          // printf("\n\ndX h2 MATRIX:\n\n");
          // for (int i = 0; i < h2_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", dX_h2_host[i]);
          // }


          // printf("\n\ndX h1 MATRIX:\n\n");
          // for (int i = 0; i < h1_size * batch_size; i++){
          //   if ((i % batch_size) == 0) {
          //     printf("\n");
          //   }
          //   printf("%f ", dX_h1_host[i]);
          // }

          printf("\n\n\n");

        }







          // UPDATE WEIGHTS + BIASES (apply learning rate and add gradients to existing params)
        float learning_rate = learning_rate_sched[cnt];

        // compute gradient clipping
        computeNorm <<< 1, 1 >>> (W_out_size, dW_out, W_out_norm);
        computeNorm <<< 1, 1 >>> (W_h3_size, dW_h3, W_h3_norm);
        computeNorm <<< 1, 1 >>> (W_h2_size, dW_h2, W_h2_norm);
        computeNorm <<< 1, 1 >>> (W_h1_size, dW_h1, W_h1_norm);

        computeNorm <<< 1, 1 >>> (output_len, dB_out, B_out_norm);
        computeNorm <<< 1, 1 >>> (h3_size, dB_h3, B_h3_norm);
        computeNorm <<< 1, 1 >>> (h2_size, dB_h2, B_h2_norm);
        computeNorm <<< 1, 1 >>> (h1_size, dB_h1, B_h1_norm);

        cudaMemcpy(W_out_norm_host, W_out_norm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h3_norm_host, W_h3_norm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h2_norm_host, W_h2_norm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h1_norm_host, W_h1_norm, sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(B_out_norm_host, B_out_norm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B_h3_norm_host, B_h3_norm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B_h2_norm_host, B_h2_norm, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B_h1_norm_host, B_h1_norm, sizeof(float), cudaMemcpyDeviceToHost);

        // FILE * W_out_norm_file = fopen("W_out_norm", "w+");
        // fwrite(W_out_norm_host, sizeof(float), size_t (1), W_out_norm_file);
        // fclose(W_out_norm_file);
        // FILE * W_h3_norm_file = fopen("W_h3_norm", "w+");
        // fwrite(W_h3_norm_host, sizeof(float), size_t (1), W_h3_norm_file);
        // fclose(W_h3_norm_file);
        // FILE * W_h2_norm_file = fopen("W_h2_norm", "w+");
        // fwrite(W_h2_norm_host, sizeof(float), size_t (1), W_h2_norm_file);
        // fclose(W_h2_norm_file);
        // FILE * W_h1_norm_file = fopen("W_h1_norm", "w+");
        // fwrite(W_h1_norm_host, sizeof(float), size_t (1), W_h1_norm_file);
        // fclose(W_h1_norm_file);

        // FILE * B_out_norm_file = fopen("B_out_norm", "w+");
        // fwrite(B_out_norm_host, sizeof(float), size_t (1), B_out_norm_file);
        // fclose(B_out_norm_file);
        // FILE * B_h3_norm_file = fopen("B_h3_norm", "w+");
        // fwrite(B_h3_norm_host, sizeof(float), size_t (1), B_h3_norm_file);
        // fclose(B_h3_norm_file);
        // FILE * B_h2_norm_file = fopen("B_h2_norm", "w+");
        // fwrite(B_h2_norm_host, sizeof(float), size_t (1), B_h2_norm_file);
        // fclose(B_h2_norm_file);
        // FILE * B_h1_norm_file = fopen("B_h1_norm", "w+");
        // fwrite(B_h1_norm_host, sizeof(float), size_t (1), B_h1_norm_file);
        // fclose(B_h1_norm_file);




        // gradientClip <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, dW_out, clipping_threshold, W_out_norm);
        // gradientClip <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, dW_h3, clipping_threshold, W_h3_norm);
        // gradientClip <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT)>>>(W_h2_size, dW_h2, clipping_threshold, W_h2_norm);
        // gradientClip <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT)>>>(W_h1_size, dW_h1, clipping_threshold, W_h1_norm);

        // gradientClip <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, dB_out, clipping_threshold, B_out_norm);
        // gradientClip <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>>(h3_size, dB_h3, clipping_threshold, B_h3_norm);
        // gradientClip <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, dB_h2, clipping_threshold, B_h2_norm);
        // gradientClip <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, dB_h1, clipping_threshold, B_h1_norm);

        // compute step sizes using second_derivs, learning rate, and safety factor (mu)
        // (converts to negative learning rate in function)...

        // first pass through store memory of old sec derivs as the first time
        if (batch_i == 0 && cnt == 0){
          cudaMemcpy(sec_dW_out_mem, sec_dW_out, W_out_size * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(sec_dW_h3_mem, sec_dW_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(sec_dW_h2_mem, sec_dW_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(sec_dW_h1_mem, sec_dW_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToDevice);

          cudaMemcpy(sec_dB_out_mem, sec_dB_out, output_len * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(sec_dB_h3_mem, sec_dB_h3, h3_size * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(sec_dB_h2_mem, sec_dB_h2, h2_size * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(sec_dB_h1_mem, sec_dB_h1, h1_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }


        // do gamma * current + (1 - gamma) * old = new. save in the current variable
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, sec_dW_out, sec_dW_out_mem, second_deriv_mem);
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, sec_dW_h3, sec_dW_h3_mem, second_deriv_mem);
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT)>>>(W_h2_size, sec_dW_h2, sec_dW_h2_mem, second_deriv_mem);
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT)>>>(W_h1_size, sec_dW_h1, sec_dW_h1_mem, second_deriv_mem);

        computeNewSecondDeriv <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, sec_dB_out, sec_dB_out_mem, second_deriv_mem);
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>>(h3_size, sec_dB_h3, sec_dB_h3_mem, second_deriv_mem);
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, sec_dB_h2, sec_dB_h2_mem, second_deriv_mem);
        computeNewSecondDeriv <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, sec_dB_h1, sec_dB_h1_mem, second_deriv_mem);

        cudaMemcpy(sec_dW_out_host, sec_dW_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sec_dW_h3_host, sec_dW_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sec_dW_h2_host, sec_dW_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sec_dW_h1_host, sec_dW_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(sec_dB_out_host, sec_dB_out, output_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sec_dB_h3_host, sec_dB_h3, h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sec_dB_h2_host, sec_dB_h2, h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sec_dB_h1_host, sec_dB_h1, h1_size * sizeof(float), cudaMemcpyDeviceToHost);

        // FILE * sec_dW_out_file = fopen("sec_dW_out", "w+");
        // fwrite(sec_dW_out_host, sizeof(float), size_t (W_out_size), sec_dW_out_file);
        // fclose(sec_dW_out_file);

        // FILE * sec_dW_h3_file = fopen("sec_dW_h3", "w+");
        // fwrite(sec_dW_h3_host, sizeof(float), size_t (W_h3_size), sec_dW_h3_file);
        // fclose(sec_dW_h3_file);

        // FILE * sec_dW_h2_file = fopen("sec_dW_h2", "w+");
        // fwrite(sec_dW_h2_host, sizeof(float), size_t (W_h2_size), sec_dW_h2_file);
        // fclose(sec_dW_h2_file);

        // FILE * sec_dW_h1_file = fopen("sec_dW_h1", "w+");
        // fwrite(sec_dW_h1_host, sizeof(float), size_t (W_h1_size), sec_dW_h1_file);
        // fclose(sec_dW_h1_file);

        // FILE * sec_dB_out_file = fopen("sec_dB_out", "w+");
        // fwrite(sec_dB_out_host, sizeof(float), size_t (output_len), sec_dB_out_file);
        // fclose(sec_dB_out_file);

        // FILE * sec_dB_h3_file = fopen("sec_dB_h3", "w+");
        // fwrite(sec_dB_h3_host, sizeof(float), size_t (h3_size), sec_dB_h3_file);
        // fclose(sec_dB_h3_file);

        // FILE * sec_dB_h2_file = fopen("sec_dB_h2", "w+");
        // fwrite(sec_dB_h2_host, sizeof(float), size_t (h2_size), sec_dB_h2_file);
        // fclose(sec_dB_h2_file);

        // FILE * sec_dB_h1_file = fopen("sec_dB_h1", "w+");
        // fwrite(sec_dB_h1_host, sizeof(float), size_t (h1_size), sec_dB_h1_file);
        // fclose(sec_dB_h1_file);

        




        // with the now new deriv we can compute step size
        computeStepSize <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, sec_dW_out, learning_rate, safety_factor, W_out_step);
        computeStepSize <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, sec_dW_h3, learning_rate, safety_factor, W_h3_step);
        computeStepSize <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT)>>>(W_h2_size, sec_dW_h2, learning_rate, safety_factor, W_h2_step);
        computeStepSize <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT)>>>(W_h1_size, sec_dW_h1, learning_rate, safety_factor, W_h1_step);

        computeStepSize <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, sec_dB_out, learning_rate, safety_factor, B_out_step);
        computeStepSize <<< SM_COUNT, ceil((float)h3_size/ SM_COUNT)>>>(h3_size, sec_dB_h3, learning_rate, safety_factor, B_h3_step);
        computeStepSize <<< SM_COUNT, ceil((float)h2_size/ SM_COUNT)>>>(h2_size, sec_dB_h2, learning_rate, safety_factor, B_h2_step);
        computeStepSize <<< SM_COUNT, ceil((float)h1_size/ SM_COUNT)>>>(h1_size, sec_dB_h1, learning_rate, safety_factor, B_h1_step);


        // save the new second deriv back to running memory
        cudaMemcpy(sec_dW_out_mem, sec_dW_out, W_out_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sec_dW_h3_mem, sec_dW_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sec_dW_h2_mem, sec_dW_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sec_dW_h1_mem, sec_dW_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemcpy(sec_dB_out_mem, sec_dB_out, output_len * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sec_dB_h3_mem, sec_dB_h3, h3_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sec_dB_h2_mem, sec_dB_h2, h2_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(sec_dB_h1_mem, sec_dB_h1, h1_size * sizeof(float), cudaMemcpyDeviceToDevice);


        // save down step sizes
        cudaMemcpy(W_out_step_host, W_out_step, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h3_step_host, W_h3_step, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h2_step_host, W_h2_step, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(W_h1_step_host, W_h1_step, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(B_out_step_host, B_out_step, output_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B_h3_step_host, B_h3_step, h3_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B_h2_step_host, B_h2_step, h2_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B_h1_step_host, B_h1_step, h1_size * sizeof(float), cudaMemcpyDeviceToHost);

        // FILE * W_out_step_file = fopen("W_out_step", "w+");
        // fwrite(W_out_step_host, sizeof(float), size_t (W_out_size), W_out_step_file);
        // fclose(W_out_step_file);

        // FILE * W_h3_step_file = fopen("W_h3_step", "w+");
        // fwrite(W_h3_step_host, sizeof(float), size_t (W_h3_size), W_h3_step_file);
        // fclose(W_h3_step_file);

        // FILE * W_h2_step_file = fopen("W_h2_step", "w+");
        // fwrite(W_h2_step_host, sizeof(float), size_t (W_h2_size), W_h2_step_file);
        // fclose(W_h2_step_file);

        // FILE * W_h1_step_file = fopen("W_h1_step", "w+");
        // fwrite(W_h1_step_host, sizeof(float), size_t (W_h1_size), W_h1_step_file);
        // fclose(W_h1_step_file);

        // FILE * B_out_step_file = fopen("B_out_step", "w+");
        // fwrite(B_out_step_host, sizeof(float), size_t (output_len), B_out_step_file);
        // fclose(B_out_step_file);

        // FILE * B_h3_step_file = fopen("B_h3_step", "w+");
        // fwrite(B_h3_step_host, sizeof(float), size_t (h3_size), B_h3_step_file);
        // fclose(B_h3_step_file);

        // FILE * B_h2_step_file = fopen("B_h2_step", "w+");
        // fwrite(B_h2_step_host, sizeof(float), size_t (h2_size), B_h2_step_file);
        // fclose(B_h2_step_file);

        // FILE * B_h1_step_file = fopen("B_h1_step", "w+");
        // fwrite(B_h1_step_host, sizeof(float), size_t (h1_size), B_h1_step_file);
        // fclose(B_h1_step_file);


        
        // multiply gradients by step size
        matScaleEls <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT)>>>(W_out_size, dW_out, W_out_step);
        matScaleEls <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT)>>>(W_h3_size, dW_h3, W_h3_step);
        matScaleEls <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT)>>>(W_h2_size, dW_h2, W_h2_step);
        matScaleEls <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT)>>>(W_h1_size, dW_h1, W_h1_step);

        matScaleEls <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, dB_out, B_out_step);
        matScaleEls <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>>(h3_size, dB_h3, B_h3_step);
        matScaleEls <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, dB_h2, B_h2_step);
        matScaleEls <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, dB_h1, B_h1_step);

        
        // add to previous parameters
        matAdd <<< SM_COUNT, ceil((float)W_out_size / SM_COUNT) >>>(W_out_size, W_out, dW_out, W_out);
        matAdd <<< SM_COUNT, ceil((float)W_h3_size / SM_COUNT) >>>(W_h3_size, W_h3, dW_h3, W_h3);
        matAdd <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT) >>>(W_h2_size, W_h2, dW_h2, W_h2);
        matAdd <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT) >>>(W_h1_size, W_h1, dW_h1, W_h1);

        matAdd <<< SM_COUNT, ceil((float)output_len / SM_COUNT)>>>(output_len, B_out, dB_out, B_out);
        matAdd <<< SM_COUNT, ceil((float)h3_size / SM_COUNT)>>>(h3_size, B_h3, dB_h3, B_h3);
        matAdd <<< SM_COUNT, ceil((float)h2_size / SM_COUNT)>>>(h2_size, B_h2, dB_h2, B_h2);
        matAdd <<< SM_COUNT, ceil((float)h1_size / SM_COUNT)>>>(h1_size, B_h1, dB_h1, B_h1);


        // RESET INTERMEDIATE MEMORY TO ZEROs (already do this within matMul, but need to reset unique Weight derivs for conv layers)
        setZero <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT) >>> (W_h2_size, dW_h2);
        setZero <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT) >>> (W_h1_size, dW_h1);
        setZero <<< SM_COUNT, ceil((float)W_h2_size / SM_COUNT) >>> (W_h2_size, sec_dW_h2);
        setZero <<< SM_COUNT, ceil((float)W_h1_size / SM_COUNT) >>> (W_h1_size, sec_dW_h1);

        // if ((cnt == 5) && (batch_i == 1002)){
        //   exit(0);
        // }


    }
    printf("\n\nITERATION RESULTS...\n\n");


    // SHOULD BE COMPUTING TRAINING STATISTICS FOR DATASET RUN HERE...

    // float totalLoss = 0;
    // float n_wrong = 0;

    // for (int k)
    // // average loss per sample in batch...
    // // not optimized...
    // computeLoss <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, Y_out, loss);
    // cudaMemcpy(loss_host, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < batch_size; i++){
    //     totalLoss += loss_host[i];
    // }

    // makePredict <<< SM_COUNT, ceil((float)batch_size / SM_COUNT)>>> (batch_size, output_len, X_out, predicted);

    // cudaMemcpy(predicted_host, predicted, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < batch_size; i++){
    //   if (predicted_host[i] != training_labels_raw[rand_batch * batch_size + i]){
    //     n_wrong++;
    //   }
    // // }
        

    printf("Avg Loss: %f\n", (float) totalLoss / training_n);
    printf("Accuracy: %f\n", (float) 1 - (n_wrong / training_n));
  }

  

  cudaEventRecord(gpu_stop);

  // output weights and biases to cpu


  // weights

  cudaMemcpy(W_h1_host, W_h1, W_h1_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h1_full_host, W_h1_full, W_h1_full_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h2_host, W_h2, W_h2_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h2_full_host, W_h2_full, W_h2_full_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_h3_host, W_h3, W_h3_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(W_out_host, W_out, W_out_size * sizeof(float), cudaMemcpyDeviceToHost);

  // biases

  cudaMemcpy(B_h1_host, B_h1, h1_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B_h2_host, B_h2, h2_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B_h3_host, B_h3, h3_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B_out_host, B_out, output_len * sizeof(float), cudaMemcpyDeviceToHost);



  cudaEventSynchronize(gpu_stop);
  float gpu_millis = 0;
  cudaEventElapsedTime(&gpu_millis, gpu_start, gpu_stop);
  printf("GPU Elapsed Millis: %f\n", gpu_millis);



  // SAVE MODEL!


  // write output to files here

  const char * model_path = "/mnt/storage/data/image_text/mnist/trained_model_softmax_singleton";

  FILE * model_file = fopen(model_path, "wb+");

  // write out weights
  fwrite(W_h1_host, sizeof(float), (size_t) W_h1_size, model_file);
  fwrite(W_h1_full_host, sizeof(float), (size_t) W_h1_full_size, model_file);  
  fwrite(W_h2_host, sizeof(float), (size_t) W_h2_size, model_file);
  fwrite(W_h2_full_host, sizeof(float), (size_t) W_h2_full_size, model_file);
  fwrite(W_h3_host, sizeof(float), (size_t) W_h3_size, model_file);
  fwrite(W_out_host, sizeof(float), (size_t) W_out_size, model_file);

  // write out biases
  fwrite(B_h1_host, sizeof(float), (size_t) h1_size, model_file);
  fwrite(B_h2_host, sizeof(float), (size_t) h2_size, model_file);
  fwrite(B_h3_host, sizeof(float), (size_t) h3_size, model_file);
  fwrite(B_out_host, sizeof(float), (size_t) output_len, model_file);

  fclose(model_file);





  // CLEANUP MEMORY...

  // FREE GPU Memory

  // free space for nodes
  cudaFree(X_in);
  cudaFree(X_h1);
  cudaFree(X_h2);
  cudaFree(X_h3);
  cudaFree(X_out);
  cudaFree(Y_out);

  // free space for values before activation (used in backprop)
  cudaFree(A_h1);
  cudaFree(A_h2);
  cudaFree(A_h3);
  cudaFree(A_out);

  // free space for transposes
  cudaFree(X_in_T);
  cudaFree(X_h1_T);
  cudaFree(X_h2_T);
  cudaFree(X_h3_T);

  // free space for sq transposed nodes
  cudaFree(sq_X_in_T);
  cudaFree(sq_X_h1_T);
  cudaFree(sq_X_h2_T);
  cudaFree(sq_W_h3_T);

  // free space for node gradients
  cudaFree(dX_h1);
  cudaFree(dX_h2);
  cudaFree(dX_h3);
  cudaFree(dX_out);

  // free space for node gradient_temporary
  cudaFree(dX_h1_activation);
  cudaFree(dX_h2_activation);
  cudaFree(dX_h3_activation);
  cudaFree(dX_out_activation);

  // free space for second deriv node stuff
  cudaFree(sec_dX_h1);
  cudaFree(sec_dX_h2);
  cudaFree(sec_dX_h3);
  cudaFree(sec_dX_out);

  cudaFree(sec_dX_h1_activation);
  cudaFree(sec_dX_h2_activation);
  cudaFree(sec_dX_h3_activation);
  cudaFree(sec_dX_out_activation);

  // free space for weights
  cudaFree(W_h1);
  cudaFree(W_h1_full);
  cudaFree(W_h2);
  cudaFree(W_h2_full);
  cudaFree(W_h3);
  cudaFree(W_out);

  // free space for weight transpose
  cudaFree(W_h1_full_T);
  cudaFree(W_h2_full_T);
  cudaFree(W_h3_T);
  cudaFree(W_out_T);

  // free space for squred weights
  cudaFree(sq_W_out_T);
  cudaFree(sq_W_h3_T);
  cudaFree(sq_W_h2_full_T);
  cudaFree(sq_W_h1_full_T);


  // free space for weight mappings...
  cudaFree(h1_null_mappings);
  cudaFree(h2_null_mappings);
  cudaFree(h1_null_set_sizes);
  cudaFree(h2_null_set_sizes);
  cudaFree(h1_full_to_unique);
  cudaFree(h1_unique_to_full);

  cudaFree(h2_full_to_unique);
  cudaFree(h2_unique_to_full);

  // free space for weight gradients
  cudaFree(dW_h1);
  cudaFree(dW_h1_full);
  cudaFree(dW_h2);
  cudaFree(dW_h2_full);
  cudaFree(dW_h3);
  cudaFree(dW_out);

  cudaFree(sec_dW_h1);
  cudaFree(sec_dW_h1_full);
  cudaFree(sec_dW_h2);
  cudaFree(sec_dW_h2_full);
  cudaFree(sec_dW_h3);
  cudaFree(sec_dW_out);

  cudaFree(sec_dW_h1_mem);
  cudaFree(sec_dW_h2_mem);
  cudaFree(sec_dW_h3_mem);
  cudaFree(sec_dW_out_mem);



  // free space for biases
  cudaFree(B_h1);
  cudaFree(B_h2);
  cudaFree(B_h3);
  cudaFree(B_out);

  // free space for bias gradients
  cudaFree(dB_h1);
  cudaFree(dB_h2);
  cudaFree(dB_h3);
  cudaFree(dB_out);

  cudaFree(sec_dB_h1);
  cudaFree(sec_dB_h2);
  cudaFree(sec_dB_h3);
  cudaFree(sec_dB_out);

  cudaFree(sec_dB_h1_mem);
  cudaFree(sec_dB_h2_mem);
  cudaFree(sec_dB_h3_mem);
  cudaFree(sec_dB_out_mem);


  // free space for storing loss values per batch
  cudaFree(loss);
  cudaFree(predicted);


  cudaFree(W_out_norm);
  cudaFree(W_h3_norm);
  cudaFree(W_h2_norm);
  cudaFree(W_h1_norm);

  cudaFree(B_out_norm);
  cudaFree(B_h3_norm);
  cudaFree(B_h2_norm);
  cudaFree(B_h1_norm);


  
  /*
  
  // Test GPU GEMM Kernel speed vs cpu...

  // PREFORM MATRIX-MATRIX on CPU
  clock_t cpu_start, cpu_end;
  float cpu_millis;
  cpu_start = clock();
  cpuMatrixMatrixMult(M, K, N, mat_left, mat_right, out);
  cpu_end = clock();
  cpu_millis = (((float) (cpu_end - cpu_start)) / CLOCKS_PER_SEC) * 1000;
  printf("CPU Elapsed Millis: %f\n", cpu_millis);
  
  */


  free(training_labels_raw);

  free(training_images);
  free(training_labels);

  free(X_in_host);
  free(Y_out_host);

  // intermediate checking values
  free(X_h1_host);
  free(X_h2_host);
  free(X_h3_host);
  free(X_out_host);
  free(X_h3_T_host);
  free(dX_h1_host);
  free(dX_h2_host);
  free(dX_h3_host);
  free(dX_out_host);
  free(dX_h1_activation_host);
  free(dX_h2_activation_host);
  free(dX_h3_activation_host);
  free(dX_out_activation_host);

  free(W_h1_host);
  free(W_h1_full_host);
  free(W_h2_host);
  free(W_h2_full_host);
  free(W_h2_full_T_host);
  free(W_h3_host);
  free(W_out_host);

  free(h1_null_mappings_host);
  free(h2_null_mappings_host);
  free(h1_null_set_sizes_host);
  free(h2_null_set_sizes_host);
  free(h1_full_to_unique_host);
  free(h1_unique_to_full_host);
  free(h2_full_to_unique_host);
  free(h2_unique_to_full_host);

  free(B_h1_host);
  free(B_h2_host);
  free(B_h3_host);
  free(B_out_host);

  

  free(loss_host);
  free(predicted_host);

  free(W_out_norm_host);
  free(W_h3_norm_host);
  free(W_h2_norm_host);
  free(W_h1_norm_host);

  free(B_out_norm_host);
  free(B_h3_norm_host);
  free(B_h2_norm_host);
  free(B_h1_norm_host);



  free(sec_dW_out_host);
  free(sec_dW_h3_host);
  free(sec_dW_h2_host);
  free(sec_dW_h1_host);

  free(sec_dB_out_host);
  free(sec_dB_h3_host);
  free(sec_dB_h2_host);
  free(sec_dB_h1_host);

  free(W_out_step_host);
  free(W_h3_step_host);
  free(W_h2_step_host);
  free(W_h1_step_host);

  free(B_out_step_host);
  free(B_h3_step_host);
  free(B_h2_step_host);
  free(B_h1_step_host);


  free(A_h1_host);
  free(A_h2_host);
  free(A_h3_host);
  free(A_out_host);

  free(softmax_jacobian_host);

  free(sec_dX_out_activation_host);
  free(sec_dX_h3_activation_host);
  free(sec_dX_h2_activation_host);
  free(sec_dX_h1_activation_host);

}
