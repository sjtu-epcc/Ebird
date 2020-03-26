#pragma once
#include <assert.h>
#include <layer/base_network_layer.h>
#include <layer/base_structure_layer.h>
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

#define index(n, c, h, w, C, H, W) ((((n) * (C) + (c)) * (H) + (h)) * (W) + (w))

namespace ebird {

template <class value_type>
class PaddingLayer : BaseNetworkLayer<value_type> {
private:
  const size_t pad_C_, pad_H_, pad_W_;

public:
  PaddingLayer(cudaStream_t layer_stream, const size_t pad_C,
               const size_t pad_H, const size_t pad_W)
      : BaseNetworkLayer<value_type>(PADDING, layer_stream),
        pad_C_(pad_C),
        pad_H_(pad_H),
        pad_W_(pad_W) {}

  void forward_setup(Registry<value_type> *reg,
                     cudnnHandle_t *cudnn_h) override;

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t *cublas_h,
                                  cudnnHandle_t *cudnn_h,
                                  Registry<value_type> *reg) override;

  void gen_description(char *buff, size_t *len_in_byte) {
    size_t meta_len_in_byte;
    size_t SIZE = sizeof(value_type);

    this->gen_meta_description(buff, &meta_len_in_byte);

    value_type pc = pad_C_, ph = pad_H_, pw = pad_W_;

    memcpy(buff + meta_len_in_byte, &pc, SIZE);
    memcpy(buff + meta_len_in_byte + 1 * SIZE, &ph, SIZE);
    memcpy(buff + meta_len_in_byte + 2 * SIZE, &pw, SIZE);

    *len_in_byte = meta_len_in_byte + 3 * SIZE;
  }

  inline cudaStream_t &get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
