/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:09:08 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <layer/padding_layer.h>

namespace ebird {

// gpu function //

template <class value_type>
void padding_forward(size_t N, size_t C, size_t H, size_t W, size_t pad_C,
                     size_t pad_H, size_t pad_W, const value_type *src,
                     value_type *dst, cudaStream_t &cur_stream);

template <class value_type>
void PaddingLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                             cudnnHandle_t *cudnn_h) {
  // hook the output of previous layer
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  assert(input != nullptr);
#ifndef NDEBUG
  printf("======>setup the forward padding layer:%d\n", this->get_id());
#endif
  size_t output_tensor_dim[4] = {input->get_N(), input->get_C() + 2 * pad_C_,
                                 input->get_H() + 2 * pad_H_,
                                 input->get_W() + 2 * pad_W_};

  Tensor<value_type> *f_out = new Tensor<value_type>(
      output_tensor_dim[0], output_tensor_dim[1], output_tensor_dim[2],
      output_tensor_dim[3], reg->get_vector(), DATA, this->get_id());

  // setup the output tensor
  this->set_f_out(f_out, reg);

  // forward hookup check
  assert(this->get_f_out() != nullptr);

  // register the forward dependency
  input = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *output = this->get_f_out();
  reg->register_forward_dependency(this->get_id(), input);
  reg->register_forward_dependency(this->get_id(), output);
}

template <class value_type>
std::vector<value_type> PaddingLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *output = this->get_f_out();

  size_t N = input->get_N(), C = input->get_C(), H = input->get_H(),
         W = input->get_W();

  padding_forward(N, C, H, W, pad_C_, pad_H_, pad_W_, input->get_gpu_ptr(),
                  output->get_gpu_ptr(), this->get_stream());

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(PaddingLayer);
}  // namespace ebird
