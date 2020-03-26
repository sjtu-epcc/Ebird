#pragma once
#include <layer/base_layer.h>
#include <registry.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class BaseStructureLayer : BaseLayer<value_type> {
private:
  std::vector<Tensor<value_type>*> inputs_;   // JOIN layer inputs
  std::vector<Tensor<value_type>*> outputs_;  // FORK layer outputs
  // std::vector<Tensor<value_type>* > b_data;  //FORK layer has one, while
  // JOIN has multiples
  StructureType type_;

public:
  BaseStructureLayer(LAYER lt, cudaStream_t layer_stream)
      : BaseLayer<value_type>(lt, layer_stream) {}

  inline int get_id() { return this->get_base_id(); }

  virtual std::vector<value_type> forward(NetworkStage stage,
                                          cublasHandle_t* cublas_h,
                                          cudnnHandle_t* cudnn_h,
                                          Registry<value_type>* reg) = 0;
  virtual void forward_setup(Registry<value_type>* reg = nullptr,
                             cudnnHandle_t* cudnn_h = nullptr) = 0;
  virtual void gen_description(char* buff, size_t* len_in_byte) = 0;

  void gen_meta_description(char* buff, size_t* len_in_byte) {
    this->_gen_meta_description(buff, len_in_byte);
  }

  std::vector<std::pair<int, int>> get_inputs_keys() {
    std::vector<BaseLayer<value_type>*> prev_layers = this->get_prev();
    int curt_l_id = this->get_base_id();
    std::vector<std::pair<int, int>> result;

    for (size_t i = 0; i < prev_layers.size(); i++) {
      int prev_id = prev_layers[i]->get_base_id();
      LayerKey key(prev_id, curt_l_id);
      result.push_back(key);
    }
    // verification
    if (type_ == FORK) {
      assert(result.size() <= 1);  // single input
    } else {
      assert(result.size() >= 1);
    }
    return result;
  }

  std::vector<std::pair<int, int>> get_outputs_keys() {
    std::vector<BaseLayer<value_type>*> next_layers = this->get_next();
    int curt_l_id = this->get_base_id();
    std::vector<std::pair<int, int>> result;

    if (next_layers.size() == 0) {
      LayerKey key(curt_l_id, 0);
      result.push_back(key);
    } else {
      for (size_t i = 0; i < next_layers.size(); i++) {
        int next_id = next_layers[i]->get_base_id();
        LayerKey key(curt_l_id, next_id);
        result.push_back(key);
      }
    }

    // verification
    if (type_ == FORK) {
      assert(result.size() >= 1);  // multiple outputs
    } else {
      assert(result.size() <= 1);
    }
    return result;
  }

  void set_input(Tensor<value_type>* t) {
    // the input a FORK layer has to be one
    if (type_ == FORK) {
      inputs_.push_back(t);
      assert(inputs_.size() <= 1);
    } else {
      inputs_.push_back(t);
    }
    if (inputs_.size() == 0) {
      return;
    } else {
      Tensor<value_type>* tmp = inputs_[0];
      assert(tmp->get_N() == t->get_N());
      // we don't check the channel size
      // assert(tmp->get_C() == t->get_C());
      assert(tmp->get_H() == t->get_H());
      assert(tmp->get_W() == t->get_W());
    }
  }

  // by convention, each layer only holds the output tensors
  void set_output(Tensor<value_type>* t, std::pair<int, int> idx,
                  Registry<value_type>* reg) {
    // the output of a FORK layer has to be one
    if (type_ == JOIN) {
      outputs_.push_back(t);
      assert(outputs_.size() <= 1);
    } else {
      outputs_.push_back(t);
    }
    reg->register_output(idx.first, idx.second, t);
  }

  // void set_b_data(Tensor<value_type>* t, std::pair<int, int> idx,
  // Registry<value_type>* reg) {
  //     //the output of a FORK layer has to be one
  //     if(type_ == JOIN) {
  //         b_data.push_back(t);
  //         assert(b_data.size() >= 1);
  //         //b data is the reverse of input pair
  //     } else if(type_ == FORK) {
  //         b_data.push_back(t);
  //         assert(b_data.size() <= 1);
  //     }
  //     reg->register_b_data(idx.second, idx.first, t);
  //     printf("@layer%d curt_b_data size %lu\n", get_id(), b_data.size() );
  // }

  void set_structure_type(StructureType t) { type_ = t; }

  // std::vector<Tensor<value_type>* > get_b_data() {
  //     if (type_ == FORK) {
  //         assert(inputs.size()  <= 1);
  //         assert(outputs.size() >= 1);
  //         assert(b_data.size()  <= 1);
  //         return this->b_data;
  //     } else if(type_ == JOIN) {
  //         assert(inputs.size()  >= 1);
  //         assert(outputs.size() <= 1);
  //         assert(b_data.size()  >= 1);
  //         return this->b_data;
  //     }
  //     return std::vector<Tensor<value_type>* >();
  // }

  std::vector<Tensor<value_type>*> get_inputs() {
    if (type_ == FORK) {
      assert(inputs_.size() <= 1);
      assert(outputs_.size() >= 1);
      return this->inputs_;
    } else if (type_ == JOIN) {
      assert(inputs_.size() >= 1);
      assert(outputs_.size() <= 1);
      return this->inputs_;
    }
    return std::vector<Tensor<value_type>*>();
  }

  std::vector<Tensor<value_type>*> get_outputs() {
    if (type_ == FORK) {
      assert(inputs_.size() <= 1);
      assert(outputs_.size() >= 1);
      return this->outputs_;
    } else if (type_ == JOIN) {
      assert(inputs_.size() >= 1);
      assert(outputs_.size() <= 1);
      return this->outputs_;
    }
    return std::vector<Tensor<value_type>*>();
  }

  inline cudaStream_t& _get_stream() { return this->_get_stream_(); }
};

}  // namespace ebird
