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
#include <layer/softmax_layer.h>

namespace ebird {
//------GPU functions-----//
// template <class value_type>
// float softmax_loss(value_type *pred_gpu_ptr, value_type *label_gpu_ptr, int
// N, int C, int H, int W);

// template <class value_type>
// value_type softmax_top1_accuracy(value_type *label, value_type *predict, int
// N, int C, int H, int W); template <class value_type> value_type
// softmax_top5_accuracy(value_type *label, value_type *predict, int N, int C,
// int H, int W);
//------------------------//

template <class value_type>
void SoftmaxLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                             cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward softmax layer:%d\n", this->get_id());
#endif
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);

  Tensor<value_type> *t_out = new Tensor<value_type>(
      t_in->get_N(), t_in->get_C(), t_in->get_H(), t_in->get_W(),
      reg->get_vector(), DATA, this->get_id());
  t_out->acquireSpaceCPU(t_out->get_N() * t_out->get_C() * t_out->get_H() *
                         t_out->get_W());
  assert(t_in != nullptr);
  this->set_f_out(t_out, reg);  // use the inplace operation.
  assert(this->get_f_out() != nullptr);

  reg->set_output_data(t_out);

  // register the forward dependency
  t_in = reg->get_reg_output(input_l, curt_l);
  //    Tensor<value_type>* t_out        = this->get_f_out();
  // Tensor<value_type> *label_train = reg->get_train_label();
  // Tensor<value_type> *label_test = reg->get_infer_label();

  assert(t_in != nullptr);
  assert(t_out != nullptr);
  // assert(label_train != nullptr);

  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), t_out);
  // reg->register_forward_dependency(this->get_id(), label_train);
}

bool has_elem(int *array, int size, size_t target) {
  for (int i = 0; i < size; i++) {
    if ((size_t)array[i] == target) {
      return true;
    }
  }
  return false;
}

template <class value_type>
std::vector<value_type> SoftmaxLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  assert(cudnn_h != nullptr);
  assert(reg != nullptr);

  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);

#ifndef NDEBUG
  printf("input tensor from %d to %d\n", input_l, curt_l);
#endif

  checkCUDNN(cudnnSoftmaxForward(
      *(cudnn_h), this->softmax_alg_, this->mode_, &(this->alpha_),
      input->get_tensor_desc(), input->get_gpu_ptr(), &(this->beta_),
      this->get_f_out()->get_tensor_desc(), this->get_f_out()->get_gpu_ptr()));
  checkCudaErrors(cudaMemcpy(
      this->get_f_out()->get_cpu_ptr(), this->get_f_out()->get_gpu_ptr(),
      this->get_f_out()->get_N() * this->get_f_out()->get_C() *
          this->get_f_out()->get_H() * this->get_f_out()->get_W() *
          sizeof(value_type),
      cudaMemcpyDeviceToHost));
#ifndef NDEBUG
  printf("output tensor from %d to %d in softmax\n", input_l, curt_l);
#endif
  if (stage == NET_INFER) {
    // loss we will compute the loss
    // Tensor<value_type> *label = reg->get_infer_label();
    // Tensor<value_type> *output = this->get_f_out();
    // const value_type normalizer = (value_type)label->get_N();
    // label->GPUtoCPU();  //TO DO:this should be done on GPU
    // output->GPUtoCPU(); //TO DO:this should be done on GPU
    // value_type corr_counter = 0;

    // value_type accuracy_top1_gpu =
    // softmax_top1_accuracy(label->get_gpu_ptr(), output->get_gpu_ptr(),
    // output->get_N(), output->get_C(), output->get_H(), output->get_W());
    // value_type accuracy_top5_gpu =
    // softmax_top5_accuracy(label->get_gpu_ptr(), output->get_gpu_ptr(),
    // output->get_N(), output->get_C(), output->get_H(), output->get_W());
    // //printf("accuracy cpu:%f gpu:%f\n", accuracy_top5_cpu,
    // accuracy_top5_gpu); std::vector<value_type> accuracy;
    // accuracy.push_back(accuracy_top1_gpu);
    // accuracy.push_back(accuracy_top5_gpu);

    /*-------------------------------*/
    return std::vector<value_type>();
    // return accuracy;
  } else {
    printf("Not supported network stage at softmax_layer.cc@line 62\n");
    return std::vector<value_type>();
  }
}

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace ebird
