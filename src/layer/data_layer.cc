/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:12:44 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <layer/data_layer.h>

namespace ebird {
template <class value_type>
void DataLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                          cudnnHandle_t *cudnn_h) {
// create tensor to store data and label
#ifndef NDEBUG
  printf("======>setup the forward data layer:%d\n", this->get_id());
#endif
  reg->set_batch_size(this->get_batch_size());
  Tensor<value_type> *f_out =
      new Tensor<value_type>(this->N_, this->C_, this->H_, this->W_,
                             reg->get_vector(), DATA_SOURCE, this->get_id());

  this->set_f_out(f_out, reg);
  reg->set_infer_data(f_out);

  // register the forward dependency
  Tensor<value_type> *output = this->get_f_out();
  reg->register_forward_dependency(this->get_id(), output);
}

template <class value_type>
std::vector<value_type> DataLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  Tensor<value_type> *output = this->get_f_out();
#ifdef PRINT_DATA
  printf(
      "------------------inferencing@layer%p output%p "
      "label%p---------------------\n",
      this, output, label);
#endif
  return std::vector<value_type>();
}

INSTANTIATE_CLASS(DataLayer);
}  // namespace ebird
