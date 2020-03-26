#pragma once
#include <layer/base_network_layer.h>
#include <switch.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class ConvolutionLayer : BaseNetworkLayer<value_type> {
private:
  // conv param
  int stride_;
  int scale_h_;
  int scale_w_;
  int padding_h_;
  int padding_w_;
  int num_output_;
  int kernel_h_, kernel_w_;
  const int conv_dims_;
  const value_type one_;
  const value_type zero_;

  //    double std;
  //    double mean;
  Initializer<value_type> *weight_initializer_;
  Initializer<value_type> *bias_initializer_;

  bool is_first_forward_ = false;

  //    weight_filler_t filler_t;

  // auxiliary conv space
  size_t f_conv_buff_size_;
  Tensor<value_type> *f_conv_buff_;

  // cudnn param
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t f_conv_alg_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnDataType_t cudnn_data_type_;
  const cudnnConvolutionMode_t cudnn_conv_mode_;

  void createDesc() {
    checkCUDNN(cudnnCreateConvolutionDescriptor(&(this->conv_desc_)));
    checkCUDNN(cudnnCreateFilterDescriptor(&(this->filter_desc_)));
  }

  // cudnnConvolutionFwdAlgoPerf_t search_fwd_algo(Registry<value_type>
  // *reg, cudnnHandle_t *cudnn_h);

  // void find_best_fwd_algo(Registry<value_type> *reg, cudnnHandle_t
  // *cudnn_h);

public:
  ConvolutionLayer(cudaStream_t layer_stream, size_t num_output,
                   size_t kernel_h, size_t kernel_w, size_t stride,
                   size_t padding_h, size_t padding_w,
                   Initializer<value_type> *weight_initializer,
                   //                 weight_filler_t ft,
                   //                 double m,
                   //                 double s,
                   bool with_bias,
                   Initializer<value_type> *bias_initializer =
                       new ConstantInitializer<float>(0.0))
      : scale_h_(1),
        scale_w_(1),
        conv_dims_(2),
        one_(1),
        zero_(0),
        cudnn_conv_mode_(CUDNN_CROSS_CORRELATION),
        weight_initializer_(weight_initializer),
        bias_initializer_(bias_initializer),
        BaseNetworkLayer<value_type>(CONV, layer_stream) {
    // setup network
    this->enable_bias(with_bias);

    assert(num_output >= 1);
    assert(kernel_h >= 1);
    assert(kernel_w >= 1);
    assert(stride >= 1);
    // conv param
    this->stride_ = stride;
    this->num_output_ = num_output;
    this->kernel_h_ = kernel_h;
    this->kernel_w_ = kernel_w;
    this->padding_h_ = padding_h;
    this->padding_w_ = padding_w;

    switch (sizeof(value_type)) {
      case 2:
        cudnn_data_type_ = CUDNN_DATA_HALF;
        break;
      case 4:
        cudnn_data_type_ = CUDNN_DATA_FLOAT;
        break;
      case 8:
        cudnn_data_type_ = CUDNN_DATA_DOUBLE;
        break;
      default:
        FatalError("Unsupported data type");
    }

    // cudnn
    createDesc();
  }

  ConvolutionLayer(cudaStream_t layer_stream, size_t num_output,
                   size_t kernel_size, size_t stride, size_t padding_h,
                   size_t padding_w,
                   Initializer<value_type> *weight_initializer, bool with_bias,
                   Initializer<value_type> *bias_initializer =
                       new ConstantInitializer<float>(0.0))
      : scale_h_(1),
        scale_w_(1),
        conv_dims_(2),
        one_(1),
        zero_(0),
        cudnn_conv_mode_(CUDNN_CROSS_CORRELATION),
        weight_initializer_(weight_initializer),
        bias_initializer_(bias_initializer),
        BaseNetworkLayer<value_type>(CONV, layer_stream) {
    // setup network
    this->enable_bias(with_bias);
    assert(bias_initializer != nullptr);

    assert(num_output >= 1);
    assert(kernel_size >= 1);
    assert(stride >= 1);
    // conv param
    this->stride_ = stride;
    this->num_output_ = num_output;
    this->kernel_h_ = kernel_size;
    this->kernel_w_ = kernel_size;
    this->padding_h_ = padding_h;
    this->padding_w_ = padding_w;

    switch (sizeof(value_type)) {
      case 2:
        cudnn_data_type_ = CUDNN_DATA_HALF;
        break;
      case 4:
        cudnn_data_type_ = CUDNN_DATA_FLOAT;
        break;
      case 8:
        cudnn_data_type_ = CUDNN_DATA_DOUBLE;
        break;
      default:
        FatalError("Unsupported data type");
    }

    // cudnn
    createDesc();
  }

  ~ConvolutionLayer() {
    checkCUDNN(cudnnDestroyConvolutionDescriptor(this->conv_desc_));
    checkCUDNN(cudnnDestroyFilterDescriptor(this->filter_desc_));
  }

  void forward_setup(Registry<value_type> *reg, cudnnHandle_t *cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t *cublas_h,
                                  cudnnHandle_t *cudnn_h,
                                  Registry<value_type> *reg);

  /**
   * Meta data
   * value_type num_output;
   * value_type kernel_h, kernel_w;
   * value_type stride;
     value_type padding_h, padding_w;
     value_type weight_initializer
     Tensor: weight, bias
   */
  void gen_description(char *buff, size_t *len_in_byte) {
    size_t meta_len_in_byte, t1, t2;
    size_t SIZE = sizeof(value_type);

    this->gen_meta_description(buff, &meta_len_in_byte);

    value_type _out = num_output_, _kh = kernel_h_, _kw = kernel_w_,
               _sd = stride_, _ph = padding_h_, _pw = padding_w_,
               _init = weight_initializer_->get_type();

    memcpy(buff + meta_len_in_byte, &_out, SIZE);
    memcpy(buff + meta_len_in_byte + 1 * SIZE, &_kh, SIZE);
    memcpy(buff + meta_len_in_byte + 2 * SIZE, &_kw, SIZE);
    memcpy(buff + meta_len_in_byte + 3 * SIZE, &_sd, SIZE);
    memcpy(buff + meta_len_in_byte + 4 * SIZE, &_ph, SIZE);
    memcpy(buff + meta_len_in_byte + 5 * SIZE, &_pw, SIZE);
    memcpy(buff + meta_len_in_byte + 6 * SIZE, &_init, SIZE);

    printf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", _out, _kh, _kw, _sd, _ph,
           _pw, _init);
    printf("(%zu %zu %zu %zu) (%zu %zu %zu %zu)\n", this->get_weight()->get_N(),
           this->get_weight()->get_C(), this->get_weight()->get_H(),
           this->get_weight()->get_W(), this->get_bias()->get_N(),
           this->get_bias()->get_C(), this->get_bias()->get_H(),
           this->get_bias()->get_W());

    this->get_weight()->gen_description(buff + meta_len_in_byte + 7 * SIZE,
                                        &t1);
    this->get_bias()->gen_description(buff + meta_len_in_byte + 7 * SIZE + t1,
                                      &t2);

    *len_in_byte = meta_len_in_byte + 7 * SIZE + t1 + t2;
  }

  inline cudaStream_t &get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
