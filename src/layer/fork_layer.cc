/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:32:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/fork_layer.h>

namespace ebird {

template <class value_type>
void ForkLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                          cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward fork layer@%d\n", this->get_id());
#endif
  LayerKey input_key = (this->get_inputs_keys())[0];
  Tensor<value_type> *input =
      reg->get_reg_output(input_key.first, input_key.second);
  this->set_input(input);
  // it has to be hard copy to avoid multiple writes
  std::vector<std::pair<int, int>> output_keys = this->get_outputs_keys();

  // should not overlap tensor !!!!
  //    this->set_output( input, output_keys[0], reg);
  for (size_t i = 0; i < output_keys.size(); i++) {
    Tensor<value_type> *tmp = new Tensor<value_type>(
        input->get_N(), input->get_C(), input->get_H(), input->get_W(),
        reg->get_vector(), DATA, this->get_id());
    this->set_output(tmp, output_keys[i], reg);
  }

  // register the forward dependency
  // please be noted the input is outputs[0]
  std::vector<Tensor<value_type> *> outputs = this->get_outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    reg->register_forward_dependency(this->get_id(), outputs[i]);
  }
  reg->register_forward_dependency(this->get_id(), input);
}

template <class value_type>
std::vector<value_type> ForkLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  assert(cudnn_h != nullptr);
  assert(reg != nullptr);
  // the input layer set the output tensor in the idx 0,
  // the subsequents should copy from idx0
  // NOTE Do nothing when forking
  Tensor<value_type> *input = this->get_inputs()[0];
  std::vector<Tensor<value_type> *> outputs = this->get_outputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i]->copy(input, this->get_stream());
  }

#ifndef NDEBUG
  printf("f fork layer : %d\n", this->get_id());
  for (size_t i = 0; i < outputs.size(); i++) {
    printf("output %zu tensor %p, layer %d\n", i, outputs[i],
           outputs[i]->get_layer_id());
  }
#endif

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(ForkLayer);

}  // namespace ebird
