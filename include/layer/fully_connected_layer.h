#pragma once
#include <layer/base_network_layer.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class FullyConnectedLayer : BaseNetworkLayer<value_type> {
private:
  // cudnn setup
  const size_t output_dim_;
  const value_type one_;
  const value_type zero_;

  //    double std;
  //    double mean;
  //    weight_filler_t filler_t;
  Initializer<value_type> *weight_initializer_;
  Initializer<value_type> *bias_initializer_;

  Tensor<value_type> *bias_multiplier_ = nullptr;

  void mat_multiply(cublasHandle_t *cublas_h, int m, int n, int k,
                    cublasOperation_t TransA, cublasOperation_t TransB,
                    value_type alpha, value_type beta, value_type *A, int lda,
                    value_type *B, int ldb, value_type *C, int ldc);

public:
  FullyConnectedLayer(cudaStream_t layer_stream, size_t output_dim,
                      Initializer<value_type> *weight_initializer,
                      bool enable_bias,
                      Initializer<value_type> *bias_initializer =
                          new ConstantInitializer<float>(0.0))
      : output_dim_(output_dim),
        one_(1),
        zero_(0),
        weight_initializer_(weight_initializer),
        bias_initializer_(bias_initializer),
        BaseNetworkLayer<value_type>(FC, layer_stream) {
    this->enable_bias(enable_bias);
    if (enable_bias) assert(this->bias_initializer_ != nullptr);
  }

  ~FullyConnectedLayer() {}

  void forward_setup(Registry<value_type> *reg, cudnnHandle_t *cudnn_h);

  // CUDNN supports in-place operations on this layer
  void hook(BaseNetworkLayer<value_type> *prev = nullptr,
            BaseNetworkLayer<value_type> *next = nullptr);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t *cublas_h,
                                  cudnnHandle_t *cudnn_h,
                                  Registry<value_type> *reg);

  /**
   * Meta Data, out, init, weight, bias
   */
  void gen_description(char *buff, size_t *len_in_byte) {
    size_t meta_len_in_byte, t1, t2;
    size_t SIZE = sizeof(value_type);

    this->gen_meta_description(buff, &meta_len_in_byte);

    value_type _out = output_dim_, _init = weight_initializer_->get_type();

    memcpy(buff + meta_len_in_byte, &_out, SIZE);
    memcpy(buff + meta_len_in_byte + 1 * SIZE, &_init, SIZE);

    this->get_weight()->gen_description(buff + meta_len_in_byte + 2 * SIZE,
                                        &t1);
    this->get_bias()->gen_description(buff + meta_len_in_byte + 2 * SIZE + t1,
                                      &t2);

    *len_in_byte = meta_len_in_byte + 2 * SIZE + t1 + t2;
  }

  inline cudaStream_t &get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
