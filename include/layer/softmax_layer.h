#pragma once
#include <layer/base_network_layer.h>
#include <math.h> /* log */
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class SoftmaxLayer : BaseNetworkLayer<value_type> {
private:
  const value_type beta_;
  const value_type alpha_;
  const cudnnSoftmaxMode_t mode_;
  const cudnnSoftmaxAlgorithm_t softmax_alg_;
  BaseNetworkLayer<value_type> *label_;

public:
  SoftmaxLayer(cudaStream_t layer_stream,
               cudnnSoftmaxAlgorithm_t alg = CUDNN_SOFTMAX_ACCURATE,
               cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE)
      : softmax_alg_(alg),
        mode_(CUDNN_SOFTMAX_MODE_INSTANCE),
        alpha_(1),
        beta_(0),
        BaseNetworkLayer<value_type>(SOFTMAX, layer_stream) {}

  ~SoftmaxLayer() {}

  void forward_setup(Registry<value_type> *reg, cudnnHandle_t *cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t *cublas_h,
                                  cudnnHandle_t *cudnn_h,
                                  Registry<value_type> *reg);

  void gen_description(char *buff, size_t *len_in_byte) {
    this->gen_meta_description(buff, len_in_byte);
  }

  inline cudaStream_t &get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
