/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:15:27 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <layer/cudnn_activation_layer.h>

namespace ebird {

template <class value_type>
void ActivationLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                                cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward activation layer:%d\n", this->get_id());
#endif
  int curt_l = this->get_id();
  int input_l = this->get_input_layer_id();
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  assert(t_in != nullptr);

  checkCUDNN(cudnnCreateActivationDescriptor(&(this->act_desc_)));
  checkCUDNN(cudnnSetActivationDescriptor(this->act_desc_, this->mode_,
                                          this->p_nan_, 0));
  Tensor<value_type> *t_out = new Tensor<value_type>(
      t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(),
      reg->get_vector(), DATA, this->get_id());

  this->set_f_out(t_out, reg);
  assert(this->get_f_out() != nullptr);

  // register the forward dependency
  t_in = reg->get_reg_output(input_l, curt_l);
  t_out = this->get_f_out();
  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), t_out);
}

template <class value_type>
std::vector<value_type> ActivationLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  assert(cudnn_h != nullptr);
  assert(reg != nullptr);

  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *t_out = this->get_f_out();

#ifndef NDEBUG
  printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

  checkCUDNN(cudnnActivationForward(
      *(cudnn_h), this->act_desc_, &(this->one_), t_in->get_tensor_desc(),
      t_in->get_gpu_ptr(), &(this->zero_), t_out->get_tensor_desc(),
      t_out->get_gpu_ptr()));

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(ActivationLayer);

}  // namespace ebird
