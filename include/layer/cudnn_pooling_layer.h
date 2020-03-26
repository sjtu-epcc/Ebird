#pragma once
#include <assert.h>
#include <layer/base_network_layer.h>
#include <switch.h>
#include <tensor.h>

namespace ebird {

template <class value_type>
class PoolingLayer : BaseNetworkLayer<value_type> {
private:
  const cudnnPoolingMode_t mode_;
  const cudnnNanPropagation_t p_nan_;
  cudnnPoolingDescriptor_t pool_desc_;
  const int vertical_stride_;
  const int horizontal_stride_;
  const int kernel_height_;
  const int kernel_width_;
  const int vertical_padding_;
  const int horizontal_padding_;
  const value_type one_;
  const value_type zero_;

public:
  // hs: horizontal stride
  // vs: vertical   stride
  // kh: kernel     height
  // hw: kernel     width
  PoolingLayer(cudaStream_t layer_stream, cudnnPoolingMode_t mode,
               cudnnNanPropagation_t p_nan, int hs, int vs, int kh, int kw,
               int hp = 0, int vp = 0)
      : mode_(mode),
        p_nan_(p_nan),
        horizontal_stride_(hs),
        vertical_stride_(vs),
        kernel_height_(kh),
        kernel_width_(kw),
        one_(1),
        zero_(0),
        vertical_padding_(vp),
        horizontal_padding_(hp),
        BaseNetworkLayer<value_type>(POOL, layer_stream) {
    // ensure network is set
    checkCUDNN(cudnnCreatePoolingDescriptor(&(this->pool_desc_)));
  }
  ~PoolingLayer() {
    checkCUDNN(cudnnDestroyPoolingDescriptor(this->pool_desc_));
  }

  void forward_setup(Registry<value_type>* reg, cudnnHandle_t* cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t* cublas_h,
                                  cudnnHandle_t* cudnn_h,
                                  Registry<value_type>* reg);

  /**
   * value_type: mode, vs, hs, kh, kw, vp, hp
   */
  void gen_description(char* buff, size_t* len_in_byte) {
    size_t meta_len_in_byte;
    this->gen_meta_description(buff, &meta_len_in_byte);

    size_t SIZE = sizeof(value_type);
    value_type _mode = mode_, vs = vertical_stride_, hs = horizontal_stride_,
               kh = kernel_height_, kw = kernel_width_, vp = vertical_padding_,
               hp = horizontal_padding_;

    memcpy(buff + meta_len_in_byte, &_mode, SIZE);
    memcpy(buff + meta_len_in_byte + 1 * SIZE, &vs, SIZE);
    memcpy(buff + meta_len_in_byte + 2 * SIZE, &hs, SIZE);
    memcpy(buff + meta_len_in_byte + 3 * SIZE, &kh, SIZE);
    memcpy(buff + meta_len_in_byte + 4 * SIZE, &kw, SIZE);
    memcpy(buff + meta_len_in_byte + 5 * SIZE, &vp, SIZE);
    memcpy(buff + meta_len_in_byte + 6 * SIZE, &hp, SIZE);

    *len_in_byte = meta_len_in_byte + 7 * SIZE;

    printf("Pool: %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", _mode, vs, hs, kh, kw,
           vp, hp);
  }

  inline cudaStream_t& get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
