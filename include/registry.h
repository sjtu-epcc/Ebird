/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:32:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once

#include <tensor.h>
#include <util/common.h>

#include <map>
#include <string>
#include <vector>

namespace ebird {

template <class value_type>
class Registry {
private:
  size_t batch_size_;
  // value_type *input_gpu_ptr_;
  value_type *output_gpu_ptr_;
  size_t output_size_;
  // value_type **output_cpu_ptr_;
  // by convention, the nullptr layer is 0
  std::vector<Tensor<value_type> *> tensors_to_free_;
  // d_key_t: [src_layer, dst_layer] current layer outputs to next layer(+1),
  // forward
  std::map<LayerKey, Tensor<value_type> *> outputs_;  // the outputs of a layer

  std::map<int, Tensor<value_type> *> bias_;
  std::map<int, Tensor<value_type> *> weight_;

  std::map<int, std::vector<Tensor<value_type> *>>
      forward_dependency_;  // NOTE  input output registered together hybridly
  std::map<Tensor<value_type> *, std::vector<int>>
      forward_dependency_by_tensor_;

  cublasHandle_t cublas_handle;

  // this tracks the computation route of a unrolled DAG/network
  std::vector<std::pair<int, NetDirect>> net_infer_route_;
  std::map<int, void *> net_layers_;
  std::map<int, std::vector<Tensor<value_type> *>> tensor_by_layer_;

  // train data shall be registered as the first layer's output
  Tensor<value_type> *infer_data_tensor_;
  Tensor<value_type> *output_data_tensor_;

  bool is_included(std::vector<Tensor<value_type> *> &v, Tensor<value_type> *t);

  void print_registry(std::map<LayerKey, Tensor<value_type> *> m,
                      const char *str) {
    for (auto const &ent : m) {
      printf("%s layer[from %d to %d] %p\n", str, ent.first.first,
             ent.first.second, ent.second);
    }
  }

  void print_registry(std::map<int, Tensor<value_type> *> m, const char *str) {
    for (auto const &ent : m) {
      printf("%s layer[%d] %p\n", str, ent.first, ent.second);
    }
  }

  void print_dependency_by_tensors(
      std::map<Tensor<value_type> *, std::vector<int>> &m, NetDirect dir);

  void print_dependency(std::map<int, std::vector<Tensor<value_type> *>> &m,
                        NetDirect dir);

  void register_layer_param(int layer_id, Tensor<value_type> *t,
                            std::map<int, Tensor<value_type> *> &m,
                            const char *str);

  Tensor<value_type> *get_layer_param(int layer_id,
                                      std::map<int, Tensor<value_type> *> &m);

  void print_net_route(std::vector<std::pair<int, NetDirect>> &route) {
    printf("computation route of this network:\n");
    for (size_t i = 0; i < route.size(); i++) {
      if (route[i].second == FORWARD) {
        printf("(%d, forward)->", route[i].first);
      } else {
        printf("Only do inference, No Backward\n");
        exit(1);
      }
    }
    printf("\n");
  }

public:
  std::vector<Tensor<value_type> *> *get_forward_dependency(int layer_id) {
    typename std::map<int, std::vector<Tensor<value_type> *>>::iterator it =
        forward_dependency_.find(layer_id);
    if (it != forward_dependency_.end()) {
      return &(it->second);
    } else {
      return nullptr;
    }
  }

  std::map<int, std::vector<Tensor<value_type> *>> &get_tensor_by_layer() {
    return this->tensor_by_layer_;
  }

  std::map<Tensor<value_type> *, std::vector<int>> &get_forward_tensors() {
    return this->forward_dependency_by_tensor_;
  }

  Registry() : infer_data_tensor_(nullptr), output_data_tensor_(nullptr) {
    checkCublasErrors(cublasCreate(&cublas_handle));
  }

  ~Registry() {
    for (size_t i = 0; i < tensors_to_free_.size(); i++) {
      delete tensors_to_free_[i];
    }
  }

  void print_forward_dependency() {
    print_dependency(forward_dependency_, FORWARD);
  }

  void print_forward_dependency_by_tensor() {
    print_dependency_by_tensors(forward_dependency_by_tensor_, FORWARD);
  }

  void print_net_infer_route() { print_net_route(this->net_infer_route_); }

  void print_net_layers() {
    typename std::map<int, void *>::iterator it = net_layers_.begin();
    printf("layers of this network:\n");
    for (it = net_layers_.begin(); it != net_layers_.end(); ++it) {
      printf("layer id:%d -> %p \n", it->first, it->second);
    }
  }

  void print_tensors_by_layers();

  std::map<int, Tensor<value_type> *> *get_all_weight() {
    return &(this->weight_);
  };

  std::map<int, Tensor<value_type> *> *get_all_bias() {
    return &(this->bias_);
  };

  // Tensor<value_type> *get_infer_data() {
  //     return infer_data;
  // }

  // Tensor<value_type> *get_output_data() {
  //     return output_data;
  // }

  std::vector<Tensor<value_type> *> *get_vector() {
    return &(this->tensors_to_free_);
  }

  void register_net_infer_route(const int layer_id, const NetDirect &nc) {
    this->net_infer_route_.push_back(std::make_pair(layer_id, nc));
  }

  void register_net_layers(const int layer_id, void *b) {
    this->net_layers_.insert(std::make_pair(layer_id, b));
  }

  void register_tensors_by_layers();

  void register_forward_dependency(int layer_id, Tensor<value_type> *t);

  void register_output(int src_layer_id, int dest_layer_id,
                       Tensor<value_type> *t);

  void register_weight(int layer_id, Tensor<value_type> *t) {
    assert(t->get_type() == PARAM);
    register_layer_param(layer_id, t, weight_, "weight");
  }

  void register_bias(int layer_id, Tensor<value_type> *t) {
    assert(t->get_type() == PARAM);
    register_layer_param(layer_id, t, bias_, "bias");
  }

  std::vector<std::pair<int, NetDirect>> &get_net_infer_route() {
    return this->net_infer_route_;
  }

  std::map<int, void *> &get_net_layers() { return this->net_layers_; }

  Tensor<value_type> *get_reg_bias(int layer_id) {
    return get_layer_param(layer_id, bias_);
  }

  Tensor<value_type> *get_reg_weight(int layer_id) {
    return get_layer_param(layer_id, weight_);
  }

  // by convention source layer holds the tensor
  Tensor<value_type> *get_reg_output(int source_layer_id, int dest_layer_id);

  size_t set_batch_size(size_t batch_size) { this->batch_size_ = batch_size; }

  size_t get_batch_size() { return this->batch_size_; }

  void set_infer_data(Tensor<value_type> *t) { this->infer_data_tensor_ = t; }

  void set_output_data(Tensor<value_type> *t) {
    this->output_data_tensor_ = t;
    this->output_gpu_ptr_ = t->get_gpu_ptr();
    this->output_size_ = t->get_C() * t->get_H() * t->get_W();
  }
  size_t get_output_size() { return this->output_size_; }
  void push_input(value_type *&input_gpu_ptr) {
    DLOG(INFO) << __FILE__ << " " << __LINE__;
    // DLOG(INFO) << "---------------------FILE_NAME: " << __FILE__
    //            << "-------------LINE: " << __LINE__ << "----------";
    this->infer_data_tensor_->replace_gpu_ptr_without_free(input_gpu_ptr);
  }

  void pull_output(value_type **&output_cpu_ptr, cudaStream_t &output_stream) {
    for (size_t i = 0; i < this->batch_size_; i++) {
      checkCudaErrors(cudaMemcpyAsync(
          output_cpu_ptr[i], this->output_gpu_ptr_ + i * this->output_size_,
          this->output_size_, cudaMemcpyDeviceToHost, output_stream));
    }
    checkCublasErrors(cudaStreamSynchronize(output_stream));
  }
};

}  // namespace ebird
