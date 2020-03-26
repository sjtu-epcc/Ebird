/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:11:21 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/join_layer.h>

namespace ebird {

template <class value_type>
void JoinLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                          cudnnHandle_t *cudnn_h) {
                                            #ifndef NDEBUG
  printf("======>setup the forward join layer@%d\n", this->get_id());
  #endif
  std::vector<std::pair<int, int>> input_keys = this->get_inputs_keys();
  for (size_t i = 0; i < input_keys.size(); i++) {
    std::pair<int, int> input_key = input_keys[i];
    Tensor<value_type> *input =
        reg->get_reg_output(input_key.first, input_key.second);
    assert(input != nullptr);
    this->set_input(input);
  }

  // join layer only has 1 output
  // reduce all the inputs into input[0]

  // should not overlap tensor !!!
  Tensor<value_type> *t = this->get_inputs()[0];
  Tensor<value_type> *output =
      new Tensor<value_type>(t->get_N(), t->get_C(), t->get_H(), t->get_W(),
                             reg->get_vector(), DATA, this->get_id());
  //    Tensor<value_type>* output = (this->get_inputs())[0];
  std::pair<int, int> output_key = (this->get_outputs_keys())[0];
  this->set_output(output, output_key, reg);
  // register the forward dependency
  std::vector<Tensor<value_type> *> inputs = this->get_inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    reg->register_forward_dependency(this->get_id(), inputs[i]);
  }
  reg->register_forward_dependency(this->get_id(), output);
}

template <class value_type>
std::vector<value_type> JoinLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  assert(cudnn_h != nullptr);
  assert(reg != nullptr);
  // we use the input tensor of idx 0 as the output,
  // the subsequents should reduce to idx 0
  std::vector<Tensor<value_type> *> inputs = this->get_inputs();
#ifndef NDEBUG
  printf("@join layer forward:\n");
  printf("input 0 : %p, layer %d\n", inputs[0], inputs[0]->get_layer_id());
#endif
  for (size_t i = 1; i < inputs.size(); i++) {
    inputs[0]->sum(inputs[i], this->get_stream());
#ifndef NDEBUG
    printf("input %zu : %p, layer %d\n", i, inputs[i],
           inputs[i]->get_layer_id());
#endif
  }
  this->get_outputs()[0]->copy(inputs[0], this->get_stream());
  // const value_type scale_factor = 1.0f/ (value_type) inputs.size();
  // inputs[0]->scale( scale_factor );

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(JoinLayer);

}  // namespace ebird
