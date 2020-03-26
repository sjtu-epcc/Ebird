/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:13:11 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/cudnn_pooling_layer.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
void PoolingLayer<value_type>::forward_setup(Registry<value_type>* reg,
                                             cudnnHandle_t* cudnn_h) {
  // hook the output of previous layer
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type>* t_in = reg->get_reg_output(input_l, curt_l);
#ifndef NDEBUG
  printf("======>setup the forward pooling layer:%d\n", this->get_id());
#endif
  checkCUDNN(cudnnSetPooling2dDescriptor(
      this->pool_desc_, this->mode_, this->p_nan_, this->kernel_height_,
      this->kernel_width_, this->vertical_padding_, this->horizontal_padding_,
      this->vertical_stride_, this->horizontal_stride_));
  int output_tensor_dim[4] = {0, 0, 0, 0};

  checkCUDNN(cudnnGetPooling2dForwardOutputDim(
      this->pool_desc_, t_in->get_tensor_desc(), &output_tensor_dim[0],
      &output_tensor_dim[1], &output_tensor_dim[2], &output_tensor_dim[3]));

  Tensor<value_type>* f_out = new Tensor<value_type>(
      output_tensor_dim[0], output_tensor_dim[1], output_tensor_dim[2],
      output_tensor_dim[3], reg->get_vector(), DATA, this->get_id());
  this->set_f_out(f_out, reg);

  // make sure all the necessary tensors are properly set
  assert(this->get_f_out() != nullptr);
  assert(t_in != nullptr);

  // register the forward dependency
  t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type>* t_out = this->get_f_out();

  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), t_out);
}

template <class value_type>
std::vector<value_type> PoolingLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t* cublas_h, cudnnHandle_t* cudnn_h,
    Registry<value_type>* reg) {
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type>* t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type>* t_out = this->get_f_out();
#ifndef NDEBUG
  printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

  checkCUDNN(cudnnPoolingForward(*(cudnn_h), this->pool_desc_, &(this->one_),
                                 t_in->get_tensor_desc(), t_in->get_gpu_ptr(),
                                 &(this->zero_), t_out->get_tensor_desc(),
                                 t_out->get_gpu_ptr()));

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace ebird
