#pragma once

#include <registry.h>
#include <tensor.h>
#include <util/common.h>

#include <vector>

namespace ebird {

/**
 * @brief  basic layer class
 * @memVar id_:layer_id; fcounter:forward_counter for join operator
 * @memFunc forward:compute forward; forward_setup:setup forward information
 */

template <class value_type>
class BaseLayer {
 private:
  static size_t instance_counter_;
  // TODO add setup for network_id
  // int network_id_; // network_id
  const int id_;         // layer_id
  size_t fcounter_ = 0;  // forward counter,  to assist join
  cudaStream_t layer_stream_;

  std::vector<BaseLayer *> next_layers_;
  std::vector<BaseLayer *> prev_layers_;

  LAYER layer_type_;

 public:
  BaseLayer(LAYER lt, cudaStream_t layer_stream)
      : id_(instance_counter_), layer_stream_(layer_stream) {
    instance_counter_++;
    this->layer_type_ = lt;
  }

  // to be implemented at each specific network/structural layers
  virtual std::vector<value_type> forward(NetworkStage stage,
                                          cublasHandle_t *cublas_h,
                                          cudnnHandle_t *cudnn_h,
                                          Registry<value_type> *reg) = 0;
  virtual void forward_setup(Registry<value_type> *reg = nullptr,
                             cudnnHandle_t *cudnn_h = nullptr) = 0;

  virtual void gen_description(char *buff, size_t *len_in_byte) = 0;

  /**
   * value_type: layer_id
   * value_type: layer_type
   * value_type: previous layers count
   * value_type: next layers count
   * value_type: [pre layer id...], [next layer id...]
   */
  void _gen_meta_description(char *buff, size_t *len_in_byte) {
    size_t SIZE = sizeof(value_type);

    value_type layer_id = id_;
    value_type type = layer_type_;
    value_type pre_cnt = prev_layers_.size();
    value_type next_cnt = next_layers_.size();

    memcpy(buff, &layer_id, SIZE);
    memcpy(buff + SIZE, &type, SIZE);
    memcpy(buff + 2 * SIZE, &pre_cnt, SIZE);
    memcpy(buff + 3 * SIZE, &next_cnt, SIZE);

    for (size_t i = 0; i < prev_layers_.size(); ++i) {
      value_type tmp = prev_layers_[i]->get_base_id();
      memcpy(buff + 4 * SIZE + i * SIZE, &tmp, SIZE);
    }
    for (size_t i = 0; i < next_layers_.size(); ++i) {
      value_type tmp = next_layers_[i]->get_base_id();
      memcpy(buff + 4 * SIZE + prev_layers_.size() * SIZE + i * SIZE, &tmp,
             SIZE);
    }

    *len_in_byte = (4 + prev_layers_.size() + next_layers_.size()) * SIZE;
  }

  inline std::vector<BaseLayer *> get_next() { return next_layers_; }

  inline std::vector<BaseLayer *> get_prev() { return prev_layers_; }

  // bi-direction hook
  void hook(BaseLayer *c) {
    if (c == nullptr) return;
#ifndef NDEBUG
    printf("hook layer %p:%d <-> %p:%d\n", this, id_, c, c->get_base_id());
#endif
    next_layers_.push_back(c);
    c->prev_layers_.push_back(this);
  }

  void switch_prev_l_to(BaseLayer *c) {
    if (c == nullptr) return;
    assert(prev_layers_.size() == 1);
    this->prev_layers_[0] = c;
#ifndef NDEBUG
    printf("switch layer %p:%d <- %p:%d\n", prev_layers_[0],
           prev_layers_[0]->get_base_id(), c, c->get_base_id());
    printf("prev layer now:%d\n", prev_layers_[0]->get_base_id());
#endif
  }

  // one direction hook
  void hook_to(BaseLayer *c) {
    if (c == nullptr) return;
#ifndef NDEBUG
    printf("hook layer %p:%d -> %p:%d\n", this, id_, c, c->get_base_id());
#endif
    next_layers_.push_back(c);
  }

  inline size_t get_fcounter() { return fcounter_; }

  inline size_t get_prev_size() { return prev_layers_.size(); }

  inline size_t get_next_size() { return next_layers_.size(); }

  inline int get_base_id() { return this->id_; }

  inline void fcounter_inc() { this->fcounter_ = this->fcounter_ + 1; }

  inline void reset_fc_counter() { this->fcounter_ = 0; }

  inline LAYER get_layer_type() { return this->layer_type_; }

  inline cudaStream_t &_get_stream_() { return this->layer_stream_; }

  inline size_t get_layer_num() { return this->instance_counter_; }
};

}  // namespace ebird
