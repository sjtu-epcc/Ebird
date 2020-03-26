#pragma once
#include <layer/base_network_layer.h>
#include <registry.h>
#include <tensor.h>

namespace ebird {

template <class value_type>
class DataLayer : public BaseNetworkLayer<value_type> {
private:
  size_t N_, C_, H_, W_;
  const DataMode mode_;

public:
  DataLayer(cudaStream_t layer_stream, DataMode m, int N, int C, int H, int W)
      : mode_(m),
        N_(N),
        C_(C),
        H_(H),
        W_(W),
        BaseNetworkLayer<value_type>(DATA_L, layer_stream) {}

  ~DataLayer() {}

  size_t get_batch_size() { return this->N_; }

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
