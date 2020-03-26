#pragma once
#include <assert.h>
#include <layer/base_network_layer.h>
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class BatchNormalizationLayer : BaseNetworkLayer<value_type> {
private:
  const value_type one_;
  const value_type zero_;
  const value_type negative_one_;

  double iter_;
  const double epsilon_;
  const cudnnBatchNormMode_t mode_;

  // forward tensor_ts
  Tensor<value_type>* resultRunningMean_;
  Tensor<value_type>* resultRunningVariance_;
  Tensor<value_type>* resultSaveMean_;
  Tensor<value_type>* resultSaveInvVariance_;

public:
  BatchNormalizationLayer(cudaStream_t layer_stream, cudnnBatchNormMode_t m,
                          double eps)
      : mode_(m),
        one_(1.0f),
        zero_(0.0f),
        iter_(1),
        epsilon_(eps),
        negative_one_(-1.0f),
        BaseNetworkLayer<value_type>(BN, layer_stream) {}

  ~BatchNormalizationLayer() {}

  void forward_setup(Registry<value_type>* reg, cudnnHandle_t* cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t* cublas_h,
                                  cudnnHandle_t* cudnn_h,
                                  Registry<value_type>* reg);

  /**
   * Meta: meta description
   * value_type: epsilon
   * value_type: BN mode
   * Tensor: weight gamma
   * Tensor: bias beta
   */
  void gen_description(char* buff, size_t* len_in_byte) {
    size_t meta_len_in_byte, t1, t2;

    this->gen_meta_description(buff, &meta_len_in_byte);

    size_t SIZE = sizeof(value_type);
    value_type eps = epsilon_;
    value_type bn_mode = mode_;

    memcpy(buff + meta_len_in_byte, &eps, SIZE);
    memcpy(buff + meta_len_in_byte + SIZE, &mode_, SIZE);

    this->get_weight()->gen_description(buff + meta_len_in_byte + 2 * SIZE,
                                        &t1);
    this->get_bias()->gen_description(buff + meta_len_in_byte + 2 * SIZE + t1,
                                      &t2);

    *len_in_byte = meta_len_in_byte + 2 * SIZE + t1 + t2;
  }

  inline cudaStream_t& get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
