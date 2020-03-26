/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:12:20 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/dropout_layer.h>

namespace ebird {

template <class value_type>
void DropoutLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                             cudnnHandle_t *cudnn_h) {
  // hook the output of previous layer
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  assert(input != nullptr);
#ifndef NDEBUG
  printf("======>setup the forward drop out layer:%d\n", this->get_id());
#endif
  checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_h, &state_size_bytes_));
  size_t state_size = state_size_bytes_ / sizeof(value_type);
  dropout_state_ = new Tensor<value_type>(
      state_size, 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
  unsigned long seed = (unsigned long)rand();
  checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc_, *cudnn_h, dropout_rate_,
                                       dropout_state_->get_gpu_ptr(),
                                       state_size_bytes_, seed));

  checkCUDNN(cudnnDropoutGetReserveSpaceSize(input->get_tensor_desc(),
                                             &buff_size_bytes_));
  size_t buff_size = buff_size_bytes_ / sizeof(value_type);
  dropout_buff_ = new Tensor<value_type>(buff_size, 1, 1, 1, reg->get_vector(),
                                         PARAM, this->get_id());

  Tensor<value_type> *f_out = new Tensor<value_type>(
      input->get_N(), input->get_C(), input->get_H(), input->get_W(),
      reg->get_vector(), DATA, this->get_id());
  // setup the output tensor
  this->set_f_out(f_out, reg);

  // forward hookup check
  assert(this->get_f_out() != nullptr);
  assert(dropout_state_ != nullptr);
  assert(dropout_buff_ != nullptr);

  // register the forward dependency
  reg->register_forward_dependency(this->get_id(), input);
  reg->register_forward_dependency(this->get_id(), dropout_state_);
  reg->register_forward_dependency(this->get_id(), dropout_buff_);
  reg->register_forward_dependency(this->get_id(), f_out);
}

template <class value_type>
std::vector<value_type> DropoutLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *output = this->get_f_out();
#ifndef NDEBUG
  printf("@ dropout layer input tensor from %d to %d\n", input_l, curt_l);
#endif

  assert(dropout_state_->get_gpu_ptr() != nullptr);

  if (stage == NET_INFER) {
    output->copy(input, this->get_stream());
  } else {
    NO_TRAIN;
    // checkCUDNN(cudnnDropoutForward(*cudnn_h,
    //                                dropout_desc,
    //                                input->get_tensor_desc(),
    //                                input->get_gpu_ptr(),
    //                                output->get_tensor_desc(),
    //                                output->get_gpu_ptr(),
    //                                dropout_buff->get_gpu_ptr(),
    //                                buff_size_bytes));
  }

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(DropoutLayer);

}  // namespace ebird
