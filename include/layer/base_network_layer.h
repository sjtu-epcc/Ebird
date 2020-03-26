#pragma once
#include <layer/base_layer.h>
#include <registry.h>
#include <tensor.h>
#include <util/common.h>
// #include <util/ebird_math.h>

namespace ebird {

template <class value_type>
class BaseNetworkLayer : BaseLayer<value_type> {
private:
  // forward output tensor, each layer only holds the output tensor
  /*----------data---------*/
  Tensor<value_type> *f_out_ = nullptr;
  // Tensor<value_type>* b_data             = nullptr;
  /*-------parameters------*/
  Tensor<value_type> *weight_ = nullptr;
  Tensor<value_type> *bias_ = nullptr;

  bool use_bias_;

public:
  BaseNetworkLayer(LAYER lt, cudaStream_t layer_stream)
      : use_bias_(false), BaseLayer<value_type>(lt, layer_stream) {}

  ~BaseNetworkLayer() {}

  int get_input_layer_id() {
    // single in
    std::vector<BaseLayer<value_type> *> inputs_l = this->get_prev();
    if (inputs_l.size() == 0) {
      return 0;
    } else {
      assert(inputs_l.size() == 1);
      BaseLayer<value_type> *input_l = inputs_l[0];
      int prev_id = input_l->get_base_id();
      return prev_id;
    }
  }

  int get_output_layer_id() {
    // single out
    std::vector<BaseLayer<value_type> *> outputs_l = this->get_next();
    if (outputs_l.size() == 0) {
      return 0;
    } else {
      assert(outputs_l.size() == 1);
      BaseLayer<value_type> *output_l = outputs_l[0];
      int prev_id = output_l->get_base_id();
      return prev_id;
    }
  }

  // needs setup before forward
  virtual std::vector<value_type> forward(NetworkStage stage,
                                          cublasHandle_t *cublas_h,
                                          cudnnHandle_t *cudnn_h,
                                          Registry<value_type> *reg) = 0;
  virtual void forward_setup(Registry<value_type> *reg = nullptr,
                             cudnnHandle_t *cudnn_h = nullptr) = 0;
  virtual void gen_description(char *buff, size_t *len_in_byte) = 0;

  void gen_meta_description(char *buff, size_t *len_in_byte) {
    this->_gen_meta_description(buff, len_in_byte);
  }

  // GET
  inline bool is_bias_enable() { return this->use_bias_; }

  inline int get_id() { return this->get_base_id(); }

  inline Tensor<value_type> *get_f_out() { return this->f_out_; }

  inline Tensor<value_type> *get_bias() { return this->bias_; }

  inline Tensor<value_type> *get_weight() { return this->weight_; }

  inline void enable_bias(bool s) { this->use_bias_ = s; }

  inline void set_f_out(Tensor<value_type> *t, Registry<value_type> *reg) {
    int cur_layer_id = this->get_id();
    int dst_layer_id = get_output_layer_id();
    reg->register_output(cur_layer_id, dst_layer_id, t);
    this->f_out_ = t;
    assert(this->get_f_out() ==
           reg->get_reg_output(cur_layer_id, dst_layer_id));
  }

  inline void set_bias(Tensor<value_type> *t, Registry<value_type> *reg) {
    this->bias_ = t;
    int cur_layer_id = this->get_id();
    reg->register_bias(cur_layer_id, t);
    assert(this->get_bias() == reg->get_reg_bias(cur_layer_id));
  }

  inline void set_weight(Tensor<value_type> *t, Registry<value_type> *reg) {
    int cur_layer_id = this->get_id();
    reg->register_weight(cur_layer_id, t);
    this->weight_ = t;
    assert(this->get_weight() == reg->get_reg_weight(cur_layer_id));
  }
  inline cudaStream_t &_get_stream() { return this->_get_stream_(); }
};

}  // namespace ebird
