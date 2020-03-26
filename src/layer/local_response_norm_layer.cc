/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:09:39 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/local_response_norm_layer.h>

namespace ebird {

template <class value_type>
void LrnLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                         cudnnHandle_t *cudnn_h) {
  // hook the output of previous layer
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  assert(t_in != nullptr);
#ifndef NDEBUG
  printf("======>setup the forward local response normalization layer:%d\n",
         this->get_id());
#endif
  checkCUDNN(cudnnSetLRNDescriptor(this->norm_desc_, this->lrn_n_,
                                   this->lrn_alpha_, this->lrn_beta_,
                                   this->lrn_k_));

  Tensor<value_type> *t_out = new Tensor<value_type>(
      t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(),
      reg->get_vector(), DATA, this->get_id());

  // setup the output tensor
  this->set_f_out(t_out, reg);

  // forward hookup check
  assert(this->get_f_out() != nullptr);

  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), t_out);
}

template <class value_type>
std::vector<value_type> LrnLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *output = this->get_f_out();
#ifndef NDEBUG
  printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

  assert(input->get_gpu_ptr() != nullptr);
  assert(output->get_gpu_ptr() != nullptr);

  checkCUDNN(cudnnLRNCrossChannelForward(
      *cudnn_h, this->norm_desc_, this->LRN_mode_, &(this->one_),
      input->get_tensor_desc(), input->get_gpu_ptr(), &(this->zero_),
      output->get_tensor_desc(), output->get_gpu_ptr()));

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(LrnLayer);

}  // namespace ebird
