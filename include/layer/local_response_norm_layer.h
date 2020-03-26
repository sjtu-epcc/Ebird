#pragma once
#include <assert.h>
#include <layer/base_network_layer.h>
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class LrnLayer : BaseNetworkLayer<value_type> {
private:
  cudnnLRNDescriptor_t norm_desc_;
  cudnnLRNMode_t LRN_mode_;
  const value_type one_;
  const value_type zero_;
  const unsigned lrn_n_;
  const double lrn_alpha_;
  const double lrn_beta_;
  const double lrn_k_;
  /*
   * Create an instance of LRN (Local Response Normalization) descriptor
   * Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from
   * Krizhevsky'12 ImageNet paper
   */

public:
  LrnLayer(cudaStream_t layer_stream)
      : one_(1),
        zero_(0),
        LRN_mode_(CUDNN_LRN_CROSS_CHANNEL_DIM1),
        lrn_n_(5.0f),
        lrn_alpha_(0.0001f),
        lrn_beta_(0.75f),
        lrn_k_(1.0f),
        BaseNetworkLayer<value_type>(LRN, layer_stream) {
    // ensure network is set
    checkCUDNN(cudnnCreateLRNDescriptor(&(this->norm_desc_)));
  }
  ~LrnLayer() { checkCUDNN(cudnnDestroyLRNDescriptor(this->norm_desc_)); }

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
