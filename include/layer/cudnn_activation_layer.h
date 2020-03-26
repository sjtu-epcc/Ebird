#pragma once
#include <assert.h>
#include <layer/base_network_layer.h>
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class ActivationLayer : BaseNetworkLayer<value_type> {
private:
  // cudnn setup
  const value_type zero_;
  const value_type one_;
  const cudnnActivationMode_t mode_;
  const cudnnNanPropagation_t p_nan_;
  cudnnActivationDescriptor_t act_desc_;

public:
  ActivationLayer(cudaStream_t layer_stream,
                  cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU,
                  cudnnNanPropagation_t p_nan = CUDNN_NOT_PROPAGATE_NAN)
      : mode_(mode),
        one_(1),
        zero_(0),
        p_nan_(p_nan),
        BaseNetworkLayer<value_type>(ACT, layer_stream) {}

  ~ActivationLayer() {
    checkCUDNN(cudnnDestroyActivationDescriptor(this->act_desc_));
  }

  void forward_setup(Registry<value_type>* reg, cudnnHandle_t* cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t* cublas_h,
                                  cudnnHandle_t* cudnn_h,
                                  Registry<value_type>* reg);

  void gen_description(char* buff, size_t* len_in_byte) {
    size_t meta_len_in_byte;
    this->gen_meta_description(buff, &meta_len_in_byte);

    //        typedef enum
    //        {
    //            CUDNN_ACTIVATION_SIGMOID      = 0,
    //            CUDNN_ACTIVATION_RELU         = 1,
    //            CUDNN_ACTIVATION_TANH         = 2,
    //            CUDNN_ACTIVATION_CLIPPED_RELU = 3
    //        } cudnnActivationMode_t;

    value_type _m = mode_;
    memcpy(buff + meta_len_in_byte, &_m, sizeof(value_type));

    *len_in_byte = meta_len_in_byte + sizeof(value_type);
  }

  inline cudaStream_t& get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
