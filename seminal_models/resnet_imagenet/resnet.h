#include <stddef.h>
#include <stdint.h>

typedef struct{
	char ** labels;
	char ** synsets;
	int * counts;
	int n_classes;
} Class_Metadata;

typedef struct {
	// starts as 224 in resnet-50
	int input;
	// 7 in resnet-50
	int init_kernel_dim;
	// 64 in resnet-50
	int init_conv_filters;
	// 2 in resnet-50
	int init_conv_stride;
	// 3 in resnet-50
	int init_maxpool_dim;
	// 2 in resnet-50
	int init_maxpool_stride;
	// in resnet-50 16 total triples
	int n_conv_blocks;
	// in resnet-50 happens at block index 3, 7, 13
	// indicates a stride change of 2 and change double of depth sizes within block
	int * is_block_spatial_reduction;
	// 1000 classes in 2012 imagenet
	// assume average pool before fully-connected layer to output
	int final_depth;
	int output;
} Dims;


// will serve as data for sequence of 1x1, 3x3, then 1x1 filters
typedef struct {
	int incoming_filters;
	int incoming_spatial_dim;
	// number of filters in first 1x1 step
	int reduced_depth;
	// number of filters in output 1x1 step
	int expanded_depth;
	// stride is 2 for the transition between stages
	int stride;
	// kernel weights for initial 1x1 step
	float * depth_reduction;
	// kernel weights for 3x3 step
	float * bias_depth_reduction;
	BatchNorm * norm_depth_reduction;
	float * spatial;
	float * bias_spatial;
	BatchNorm * norm_spatial;
	// kernel weights for output 1x1 step 
	float * depth_expansion;
	float * bias_depth_expansion;
	// need a projection is input dims != output dims
	// contains pointers to convluations transforming input to output to add as residual
	// occurs between stages:
	// transform init max pool output (64, 56, 56) -> stage 0 output (256, 56, 56)
	// transform stage 0, last layer output (256, 56, 56) -> stage 1 output (512, 28, 28)
	// transform stage 1 output (512, 28, 28) -> stage 2 output (1024, 14, 14)
	// transform stage 2 output (1024, 14, 14) -> stage 3 output (2048, 7, 7) 
	// going from 2048 average pooled outputs of last conv layer to 1000 classes softmax
	// if depths are different than need a projection (if spatial also different then will do a 3x3, stride 2 kernel over block input)
	// if depths different, but same spatial, then do a 1x1 convolution
	float * projection;
	// will be equal to number of output filters
	float * bias_projection;
	BatchNorm * norm_residual_added;
} ConvBlock;

typedef struct{
	int spatial_dim;
	int depth;
	float * gamma;
	float * beta;
} BatchNorm;

typedef struct{
	// initial 7x7 kernels
	float * init_conv_layer;
	float * bias_init_conv;
	BatchNorm * norm_init_conv;
	// contains pointers to collection of bottleneck block triples
	ConvBlock ** conv_blocks;
	float * fully_connected;
	float ** locations;
	int * sizes;
	int n_locations;
} Params;

typedef struct {
	int incoming_filters;
	int incoming_spatial_dim;
	// number of filters in first 1x1 step
	int reduced_depth;
	// number of filters in output 1x1 step
	int expanded_depth;
	// stride 2 for transition between stages
	int stride;
	// applying first layer in block to output of previous block and adding bias
	float *post_reduced;
	Cache_BatchNorm * norm_post_reduced;
	float *post_reduced_activated;

	// applying second layer in block to depth_reduced and adding bias
	float *post_spatial;
	Cache_BatchNorm * norm_post_spatial;
	float *post_spatial_activated;

	float *post_expanded;

	// if input dim of block != output dim of block, need to apply a transform 
	// (otherwise null which implies identity of output of previous block)
	float *transformed_residual;
	// occurs after adding last layer to residual connection
	// adding transformed_residual (or equivalently input of block == output of prev block) to norm_post_expanded -> normalized
	float *output;
	Cache_BatchNorm * norm_post_residual_added;
	// doing normalization, then ReLU
	float *output_activated;
} Activation_ConvBlock;

typedef struct{
	int input_size;
	int feature_size;
	float * means;
	float * vars;
	float * normalized_temp;
	float * normalized;
} Cache_BatchNorm;

typedef struct{
	// after initial 7x7 kernel and adding bias
	float * init_conv_applied;
	Cache_BatchNorm * norm_init_conv;
	float * init_conv_activated;
	// saving max inds for backprop of maxpool layer in beginning
	int * max_inds;
	// occurs after max pool of init conv activations
	float * init_convblock_input;
	Activation_ConvBlock ** activation_conv_blocks;
	int n_conv_blocks;
	// occurs after average pool of final convblock output
	float * final_conv_output_pooled;
	// after appying fully connected layer to final_conv_output_pooled
	float * linear_output;
} Activations;

typedef struct{
	Dims * dims;
	Params * params;
} ResNet;


typedef struct{
	Activations * activations;
	// after applying softmax to linear_output
	float * pred;
	// copy to cpu
	float * pred_cpu;
} Forward_Buffer;

typedef struct{
	float * output_layer_deriv;
	Params * param_derivs;
	Params * prev_means;
	Params * prev_vars;
	Activations * activation_derivs;
} Backprop_Buffer;


typedef struct {
	ResNet * model;
	Batch * cur_batch;
	Forward_Buffer * forward_buffer;
	Backprop_Buffer * backprop_buffer;
	float learning_rate;
	float base_mean_decay;
	float base_var_decay;
	float cur_mean_decay;
	float cur_var_decay;
	float eps;
	int batch_size;
	int n_epochs;
	float * loss_per_epoch;
	float * accuracy_per_epoch;
} Train_ResNet;

typedef struct{
	int image_dim;
	int image_size;
	int n_images;
	uint8_t * images_cpu;
	float * images_float_cpu;
	float * images;
	int * correct_classes_cpu;
	int * correct_classes;
} Batch;

