/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:14:52 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <cudnn.h>
#include <layer/cudnn_convolution_layer.h>
#include <util/mem_util.h>

#include <limits>
#define CONV_DEBUG

namespace ebird {

template <class value_type>
void ConvolutionLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                                 cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward convolution layer:%d\n", this->get_id());
#endif
  assert(reg != nullptr);
  assert(cudnn_h != nullptr);

  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();

  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);

  // get the previous layer forward output
  assert(t_in != nullptr);

  const int conv_input_channels = t_in->get_C();
  const int conv_outputs = this->num_output_;
  const int conv_kernel_h = this->kernel_h_;
  const int conv_kernel_w = this->kernel_w_;
  const int conv_dims = 2;
  const int padding[2] = {this->padding_h_, this->padding_w_};
  const int upscales[conv_dims] = {this->scale_h_, this->scale_w_};
  const int strides[conv_dims] = {this->stride_, this->stride_};
  // filter description
  const int filter_dim = 4;
  const int filter_dims[filter_dim] = {(int)conv_outputs,
                                       (int)conv_input_channels,
                                       (int)conv_kernel_h, (int)conv_kernel_w};
  // create filter weight tensor
  Tensor<value_type> *weight = new Tensor<value_type>(
      conv_outputs, conv_input_channels, conv_kernel_h, conv_kernel_w,
      reg->get_vector(), PARAM, this->get_id());
  this->set_weight(weight, reg);

  weight->init(this->weight_initializer_);

  checkCUDNN(cudnnSetFilterNdDescriptor(
      this->filter_desc_, this->cudnn_data_type_, t_in->get_tensor_format(),
      filter_dim, filter_dims));

  // v 1.0 only supports image convolution
  checkCUDNN(cudnnSetConvolutionNdDescriptor(
      this->conv_desc_, this->conv_dims_, padding, strides, upscales,
      this->cudnn_conv_mode_, this->cudnn_data_type_));

  int t_output_dim[4] = {0, 0, 0, 0};
  checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
      this->conv_desc_, t_in->get_tensor_desc(), this->filter_desc_, filter_dim,
      t_output_dim));
  // create output tensor
  Tensor<value_type> *t_out = new Tensor<value_type>(
      t_output_dim[0], t_output_dim[1], t_output_dim[2], t_output_dim[3],
      reg->get_vector(), DATA, this->get_id());
  this->set_f_out(t_out, reg);
  // create bias tensor
  Tensor<value_type> *bias = new Tensor<value_type>(
      1, weight->get_N(), 1, 1, reg->get_vector(), PARAM, this->get_id());
  // TODO: bias == 0 ?!
  bias->init(this->bias_initializer_);
  this->set_bias(bias, reg);

  int conv_fwd_algo_cnt;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(*cudnn_h,
                                                         &conv_fwd_algo_cnt));
  cudnnConvolutionFwdAlgoPerf_t perf_results[conv_fwd_algo_cnt];
  // we search the fastest algorithm
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
      *cudnn_h, t_in->get_tensor_desc(), this->filter_desc_, this->conv_desc_,
      t_out->get_tensor_desc(), conv_fwd_algo_cnt, &conv_fwd_algo_cnt,
      perf_results));
  this->f_conv_alg_ = perf_results[0].algo;
  this->f_conv_buff_size_ = perf_results[0].memory;

  size_t buff_W = (this->f_conv_buff_size_) / sizeof(value_type) + 1;
  this->f_conv_buff_ = new Tensor<value_type>(
      1, 1, 1, buff_W, reg->get_vector(), CONV_BUFF, this->get_id());
#ifndef NDEBUG
  printf("\n------------------conv-layer forward setup %d \n", this->get_id());
  printf("output tensor dims:%d %d %d %d \n", t_output_dim[0], t_output_dim[1],
         t_output_dim[2], t_output_dim[3]);
  printf("Fastest forward conv is Algo %d\n", this->f_conv_alg_);
  printf("the size of forward CNN buffer space is %.3f MB\n",
         (double)this->f_conv_buff_size_ / 1024.0f / 1024.0f);
#endif
  // make sure all the necessary tensors are properly set
  assert(this->get_f_out() != nullptr);
  assert(this->get_bias() != nullptr);
  assert(this->get_weight() != nullptr);
  assert(this->f_conv_buff_ != nullptr);

  // register the forward dependency
  t_in = reg->get_reg_output(input_l, curt_l);
  weight = this->get_weight();
  t_out = this->get_f_out();

  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), weight);
  reg->register_forward_dependency(this->get_id(), t_out);
  reg->register_forward_dependency(this->get_id(), this->f_conv_buff_);

  if (this->is_bias_enable()) {
    reg->register_forward_dependency(this->get_id(), this->get_bias());
  }
}

template <class value_type>
std::vector<value_type> ConvolutionLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  assert(cudnn_h != nullptr);
  assert(reg != nullptr);

  // setup each parameters on GPUs
#ifndef NDEBUG
  double start = get_cur_time();
#endif

  // prepare the input tensors
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *weight = this->get_weight();
  Tensor<value_type> *f_out = this->get_f_out();

  assert(input != nullptr);
  assert(weight != nullptr);
  assert(f_out != nullptr);

  assert(input->get_gpu_ptr() != nullptr);
  assert(weight->get_gpu_ptr() != nullptr);
  assert(this->f_conv_buff_->get_gpu_ptr() != nullptr);
  assert(f_out->get_gpu_ptr() != nullptr);

  checkCUDNN(cudnnConvolutionForward(
      *(cudnn_h), &(this->one_), input->get_tensor_desc(), input->get_gpu_ptr(),
      this->filter_desc_, weight->get_gpu_ptr(), this->conv_desc_,
      this->f_conv_alg_, this->f_conv_buff_->get_gpu_ptr(),
      this->f_conv_buff_size_, &(this->zero_), f_out->get_tensor_desc(),
      f_out->get_gpu_ptr()));

  if (this->is_bias_enable()) {
    checkCUDNN(cudnnAddTensor(*(cudnn_h), &(this->one_),
                              this->get_bias()->get_tensor_desc(),
                              this->get_bias()->get_gpu_ptr(), &(this->one_),
                              f_out->get_tensor_desc(), f_out->get_gpu_ptr()));
  }

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace ebird
