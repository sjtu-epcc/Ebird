/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:10:56 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/fully_connected_layer.h>

namespace ebird {

template <class value_type>
void FullyConnectedLayer<value_type>::forward_setup(Registry<value_type> *reg,
                                                    cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward fully connected layer:%d\n", this->get_id());
#endif
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);

#ifndef NDEBUG
  printf("the layer's input tensor:%p, size : %zu %zu %zu %zu\n", input,
         input->get_N(), input->get_C(), input->get_H(), input->get_W());
#endif
  assert(input != nullptr);

  const size_t N = input->get_N();
  const size_t C = input->get_C();
  const size_t H = input->get_H();
  const size_t W = input->get_W();
  const size_t input_dim = C * H * W;
  const size_t ouput_dim = this->output_dim_;
  // right align rule applied to tensor
  // setup weight
  Tensor<value_type> *weight = new Tensor<value_type>(
      ouput_dim, input_dim, 1, 1, reg->get_vector(), PARAM, this->get_id());
  this->set_weight(weight, reg);
  weight->init(this->weight_initializer_);

  // setup bias
  Tensor<value_type> *bias = new Tensor<value_type>(
      output_dim_, 1, 1, 1, reg->get_vector(), PARAM, this->get_id());
  Tensor<value_type> *bias_multiplier = new Tensor<value_type>(
      1, 1, 1, N, reg->get_vector(), AUX, this->get_id());
  bias->init(this->bias_initializer_);
  // to remove
  for (size_t i = 0; i < N; i++) bias_multiplier->set_scalar(0, 0, 0, i, 1);
  // bias_multiplier->init(new ConstantInitializer<value_type>(1.0f));

  this->set_bias(bias, reg);
  this->bias_multiplier_ = bias_multiplier;

  // setup output tensor
  Tensor<value_type> *output = new Tensor<value_type>(
      N, output_dim_, 1, 1, reg->get_vector(), DATA, this->get_id());
  this->set_f_out(output, reg);

  assert(this->get_weight() != nullptr);
  assert(this->get_bias() != nullptr);
  assert(this->get_f_out() != nullptr);
  assert(this->bias_multiplier_ != nullptr);

  // register the forward dependency
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *t_out = this->get_f_out();
  bias = this->get_bias();
  weight = this->get_weight();

  assert(t_in != nullptr);
  assert(weight != nullptr);
  assert(t_out != nullptr);
  assert(bias != nullptr);

  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), weight);
  reg->register_forward_dependency(this->get_id(), t_out);
  reg->register_forward_dependency(this->get_id(), bias);
}

template <>
void FullyConnectedLayer<float>::mat_multiply(
    cublasHandle_t *cublas_h, int m, int n, int k, cublasOperation_t TransA,
    cublasOperation_t TransB, float alpha, float beta, float *A, int lda,
    float *B, int ldb, float *C, int ldc) {
  checkCublasErrors(cublasSgemm(*(cublas_h), TransA, TransB, m, n, k, &alpha, A,
                                lda, B, ldb, &beta, C, ldc));
}

template <>
void FullyConnectedLayer<double>::mat_multiply(
    cublasHandle_t *cublas_h, int m, int n, int k, cublasOperation_t TransA,
    cublasOperation_t TransB, double alpha, double beta, double *A, int lda,
    double *B, int ldb, double *C, int ldc) {
  checkCublasErrors(cublasDgemm(*(cublas_h), TransA, TransB, m, n, k, &alpha, A,
                                lda, B, ldb, &beta, C, ldc));
}

template <class value_type>
std::vector<value_type> FullyConnectedLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  //--------forward data operation--------//
  const int m_d = (int)this->get_f_out()->get_N();   // output dim
  const int k_d = (int)this->get_weight()->get_C();  //@line14, input dim
  const int n_d = (int)this->get_weight()->get_N();  //@line14, total images

  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *weight = this->get_weight();
  Tensor<value_type> *t_out = this->get_f_out();
  Tensor<value_type> *bias = this->get_bias();

#ifndef NDEBUG
  printf("input tensor from %d to %d\n", input_l, curt_l);
#endif
  // forward data
  mat_multiply(cublas_h, n_d, m_d, k_d, CUBLAS_OP_T, CUBLAS_OP_N, this->one_,
               this->zero_, weight->get_gpu_ptr(), k_d, t_in->get_gpu_ptr(),
               k_d, t_out->get_gpu_ptr(), n_d);

  //--------forward bias operation--------//
  if (this->is_bias_enable()) {
    mat_multiply(cublas_h, n_d, m_d, this->one_, CUBLAS_OP_N, CUBLAS_OP_N,
                 this->one_, this->one_, bias->get_gpu_ptr(), (int)n_d,
                 this->bias_multiplier_->get_gpu_ptr(), (int)this->one_,
                 t_out->get_gpu_ptr(), (int)n_d);
  }

  return std::vector<value_type>();
}

INSTANTIATE_CLASS(FullyConnectedLayer);

}  // namespace ebird
