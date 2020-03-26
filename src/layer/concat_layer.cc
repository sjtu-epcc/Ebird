/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:15:51 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <layer/concat_layer.h>

namespace ebird {

template <class value_type>
void ConcatLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                            cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward concat layer@%d\n", this->get_id());
#endif
  std::vector<std::pair<int, int>> input_keys = this->get_inputs_keys();
  for (size_t i = 0; i < input_keys.size(); i++) {
    std::pair<int, int> input_key = input_keys[i];
    Tensor<value_type> *input =
        reg->get_reg_output(input_key.first, input_key.second);
    assert(input != nullptr);
    this->set_input(input);
  }

  // concat layer only has 1 output
  size_t N = (this->get_inputs())[0]->get_N();
  // concat according C axis
  size_t C = 0;
  for (size_t i = 0; i < this->get_inputs().size(); ++i) {
    C += this->get_inputs()[i]->get_C();
  }
  size_t H = (this->get_inputs())[0]->get_H();
  size_t W = (this->get_inputs())[0]->get_W();

  Tensor<value_type> *output = new Tensor<value_type>(
      N, C, H, W, reg->get_vector(), DATA, this->get_id());
  this->set_output(output, (this->get_outputs_keys())[0], reg);

  // register the forward dependency
  std::vector<Tensor<value_type> *> inputs = this->get_inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    reg->register_forward_dependency(this->get_id(), inputs[i]);
  }
  reg->register_forward_dependency(this->get_id(), output);
}

template <class value_type>
std::vector<value_type> ConcatLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  assert(cudnn_h != nullptr);
  assert(reg != nullptr);
  std::vector<Tensor<value_type> *> inputs = this->get_inputs();
  Tensor<value_type> *output = (this->get_outputs())[0];

  // stack output by inputs according to C axis
  //    size_t one_input_size = inputs[0]->get_C() * inputs[0]->get_H() *
  //    inputs[0]->get_W();
  size_t offset = 0;
  for (size_t i = 0; i < inputs.size(); i++) {
    // output size is larger than input, so stack the input !
    output->copy(inputs[i], this->get_stream(), -1, -1, (int)offset,
                 (int)(offset + inputs[i]->get_N() * inputs[i]->get_C() *
                                    inputs[i]->get_H() * inputs[i]->get_W()));

    offset += inputs[i]->get_N() * inputs[i]->get_C() * inputs[i]->get_H() *
              inputs[i]->get_W();
  }
  return std::vector<value_type>();
}

INSTANTIATE_CLASS(ConcatLayer);

}  // namespace ebird
