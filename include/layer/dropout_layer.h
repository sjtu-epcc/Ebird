#pragma once
#include <assert.h>
#include <layer/base_network_layer.h>
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class DropoutLayer : BaseNetworkLayer<value_type> {
private:
  const value_type one_;
  const value_type zero_;
  const double dropout_rate_;
  size_t state_size_bytes_;
  size_t buff_size_bytes_;
  Tensor<value_type> *dropout_state_;
  Tensor<value_type> *dropout_buff_;
  cudnnDropoutDescriptor_t dropout_desc_;
  uint64_t seed = 1337ull;

public:
  DropoutLayer(cudaStream_t layer_stream, double dr)
      : one_(1),
        zero_(0),
        buff_size_bytes_(0),
        state_size_bytes_(0),
        dropout_rate_(dr),
        BaseNetworkLayer<value_type>(DROPOUT, layer_stream) {
    checkCUDNN(cudnnCreateDropoutDescriptor(&(this->dropout_desc_)));
  }
  ~DropoutLayer() {
    checkCUDNN(cudnnDestroyDropoutDescriptor(this->dropout_desc_));
  }

  void forward_setup(Registry<value_type> *reg, cudnnHandle_t *cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t *cublas_h,
                                  cudnnHandle_t *cudnn_h,
                                  Registry<value_type> *reg);

  void gen_description(char *buff, size_t *len_in_byte) {
    size_t meta_len_in_byte;
    this->gen_meta_description(buff, &meta_len_in_byte);

    value_type dr = dropout_rate_;
    memcpy(buff + meta_len_in_byte, &dr, sizeof(value_type));

    *len_in_byte = meta_len_in_byte + sizeof(value_type);
  }

  inline cudaStream_t &get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
