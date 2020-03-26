/*
 * Created Date: Thursday, February 6th 2020, 3:25:05 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:32:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2020 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <mem_reducer.h>
namespace ebird {

template <class value_type>
void MemReducer<value_type>::set_regulated_tensors() {
  std::map<int, std::vector<Tensor<value_type> *>> tensors_by_layer =
      registry_->get_tensor_by_layer();
  typename std::map<int, std::vector<Tensor<value_type> *>>::iterator it =
      tensors_by_layer.begin();
  for (it = tensors_by_layer.begin(); it != tensors_by_layer.end(); ++it) {
    std::vector<Tensor<value_type> *> tensors = it->second;
    std::vector<Tensor<value_type> *> tmp_tensors =
        std::vector<Tensor<value_type> *>();
    tmp_tensors.clear();
    for (size_t i = 0; i < tensors.size(); i++) {
      if (tensors[i]->get_type() != DATA &&
          tensors[i]->get_type() != CONV_BUFF) {
        continue;
      }
      tmp_tensors.emplace_back(tensors[i]);
    }
    regulated_tensors_by_layer_.emplace(it->first, tmp_tensors);
  }
}

template <class value_type>
void MemReducer<value_type>::init(Registry<value_type> *reg) {
  this->registry_ = reg;

  // reset containes
  tensor_gpu_ads_.clear();
  gpu_ad_sizes_.clear();
  idle_gpu_ads_.clear();
  gpu_ad_ptrs_.clear();
  regulated_tensors_by_layer_.clear();
  subsequent_forward_.clear();

  // filter tensors
  set_regulated_tensors();
  std::vector<std::pair<int, NetDirect>> net_route =
      registry_->get_net_infer_route();
  max_layer_cnt_ = net_route.size();
  for (auto layer = net_route.begin(); layer != net_route.end(); layer++) {
    std::vector<int> tmp_sub = std::vector<int>();
    for (auto sub_layer = layer + 1; sub_layer != net_route.end();
         sub_layer++) {
      tmp_sub.emplace_back(sub_layer->first);
    }
    subsequent_forward_.emplace(layer->first, tmp_sub);
  }
}

template <class value_type>
std::vector<int> &MemReducer<value_type>::get_subsequent_layers(
    int curt_layer_id) {
  auto ret_it = subsequent_forward_.find(curt_layer_id);
  if (ret_it != subsequent_forward_.end()) {
    return ret_it->second;
  } else {
    LOG(FATAL) << "Layer Not Found In Subsequent Forward";
  }
}

template <class value_type>
bool MemReducer<value_type>::is_used_by_layer(int layer_id,
                                              Tensor<value_type> *t) {
  std::vector<Tensor<value_type> *> *tensors = nullptr;
  tensors = registry_->get_forward_dependency(layer_id);
  if (tensors == nullptr) return false;
  for (size_t i = 0; i < tensors->size(); i++) {
    if (tensors->operator[](i) == t) {
      return true;
    }
  }
  return false;
}

template <class value_type>
bool MemReducer<value_type>::is_freeable_afterwards(int curt_layer_id,
                                                    Tensor<value_type> *t) {
  std::vector<int> subsequent_layers = get_subsequent_layers(curt_layer_id);
  for (size_t i = 0; i < subsequent_layers.size(); i++) {
    int tmp_layer_id = subsequent_layers[i];
    bool is_used = is_used_by_layer(tmp_layer_id, t);
    if (is_used) {
      return false;
    }
  }
  return true;
}

/**
 * @brief scan the network for topology
 *
 * @tparam value_type int float double
 * @param start_layer start layer of the network
 * @param reg registry of the network
 *
 */
template <class value_type>
void MemReducer<value_type>::scanNet() {
  /**from start to end layer
   * 1. seek a propriate ptrs for current dependency tensor from idle ptr pool
   * or allocate a new one.
   * 2. push no-need ptrs to idel ptr
   */
  std::vector<std::pair<int, NetDirect>> net_route =
      registry_->get_net_infer_route();
  int net_size = net_route.size();
#ifndef NDEBUG
  assert(net_size >= 2);
#endif
  // auto end_layer_it = std::prev(net_route.end());
  auto end_layer_it = net_route.end();
  // NOTE no input layer and output layer
  int gpu_ad_id = 0;
  for (auto it = net_route.begin() + 1; it != end_layer_it; ++it) {
    // allocate ptr for new tensors
#ifndef NDEBUG
    printf("scanning layer %d\n", it->first);
#endif
    typename std::unordered_map<
        int, std::vector<Tensor<value_type> *>>::iterator it2 =
        regulated_tensors_by_layer_.find(it->first);

    if (it2 != regulated_tensors_by_layer_.end()) {
      // typename std::vector<Tensor<value_type *>>::iterator tensor_it =
      // it2->second.begin();
      for (auto tensor_it = it2->second.begin(); tensor_it != it2->second.end();
           ++tensor_it) {
        auto if_alloc = busy_tensors_.find(*tensor_it);
        if (if_alloc != busy_tensors_.end()) continue;

        int tensor_id = (*tensor_it)->get_tensor_id();
        int tensor_size = (*tensor_it)->get_mem_size();

        auto idle_gpu_ads_it = idle_gpu_ads_.lower_bound(tensor_size);
        // TODO 找到合适的就用，找不到合适的看是否仍有
        if (idle_gpu_ads_it != idle_gpu_ads_.end()) {
          auto if_insert_tensor =
              tensor_gpu_ads_.emplace(*tensor_it, idle_gpu_ads_it->second);
#ifndef NDEBUG
          assert(if_insert_tensor.second);
#endif
          busy_tensors_.emplace(*tensor_it, idle_gpu_ads_it->second);
          idle_gpu_ads_.erase(idle_gpu_ads_it);
        } else {
          auto if_insert_tensor =
              tensor_gpu_ads_.emplace(*tensor_it, gpu_ad_id);
#ifndef NDEBUG
          assert(if_insert_tensor.second);
#endif
          busy_tensors_.emplace(*tensor_it, gpu_ad_id);
          gpu_ad_sizes_.emplace(gpu_ad_id, tensor_size);
          gpu_ad_id++;
        }
      }
    }
#ifndef NDEBUG
    printf("------------------adding--------------------\n");
    printf("------------------busy_tensors--------------------\n");

    for (auto debug_it : busy_tensors_) {
      printf("tensor: %p,GPU address id: %d\n", debug_it.first,
             debug_it.second);
    }
    printf("------------------idle_gpu_ads--------------------\n");
    for (auto debug_it : idle_gpu_ads_) {
      printf("gpu address size: %d, gpu address id: %d\n", debug_it.first,
             debug_it.second);
    }
    printf("------------------tensor_gpu_ads--------------------\n");
    for (auto debug_it : tensor_gpu_ads_) {
      printf("tensor: %p, GPU address id: %d\n", debug_it.first,
             debug_it.second);
    }
    printf("------------------gpu_ad_sizes--------------------\n");
    for (auto debug_it : gpu_ad_sizes_) {
      printf("gpu address id: %d, gpu address size: %d\n", debug_it.first,
             debug_it.second);
    }
#endif

    // remove idle tensors
    typename std::unordered_map<Tensor<value_type> *, int>::iterator busy_it =
        busy_tensors_.begin();
    for (; busy_it != busy_tensors_.end();) {
      if (is_freeable_afterwards(it->first, busy_it->first)) {
        auto gpu_ad_size = gpu_ad_sizes_.find(busy_it->second);
#ifndef NDEBUG
        assert(gpu_ad_size != gpu_ad_sizes_.end());
#endif
        idle_gpu_ads_.emplace(gpu_ad_size->second, gpu_ad_size->first);
        busy_it = busy_tensors_.erase(busy_it);
      } else {
        busy_it++;
      }
    }
#ifndef NDEBUG
    printf("------------------removing--------------------\n");
    printf("------------------busy_tensors--------------------\n");

    for (auto debug_it : busy_tensors_) {
      printf("tensor: %p,GPU address id: %d\n", debug_it.first,
             debug_it.second);
    }
    printf("------------------idle_gpu_ads--------------------\n");
    for (auto debug_it : idle_gpu_ads_) {
      printf("gpu address size: %d, gpu address id: %d\n", debug_it.first,
             debug_it.second);
    }
    printf("------------------tensor_gpu_ads--------------------\n");
    for (auto debug_it : tensor_gpu_ads_) {
      printf("tensor: %p, GPU address id: %d\n", debug_it.first,
             debug_it.second);
    }
    printf("------------------gpu_ad_sizes--------------------\n");
    for (auto debug_it : gpu_ad_sizes_) {
      printf("gpu address id: %d, gpu address size: %d\n", debug_it.first,
             debug_it.second);
    }
#endif
  }
}

template <class value_type>
void MemReducer<value_type>::gpuAlloc() {
  for (auto it = gpu_ad_sizes_.begin(); it != gpu_ad_sizes_.end(); ++it) {
    value_type *tmp_gpu_ptr;
    checkCudaErrors(cudaMalloc(&tmp_gpu_ptr, it->second));
    gpu_ad_ptrs_.emplace(it->first, tmp_gpu_ptr);
  }
  for (auto it = tensor_gpu_ads_.begin(); it != tensor_gpu_ads_.end(); ++it) {
    Tensor<value_type> *tmp_tensor = it->first;
    int tmp_gpu_ad_id = it->second;

    typename std::unordered_map<int, value_type *>::iterator gpu_ad_it =
        gpu_ad_ptrs_.find(tmp_gpu_ad_id);
#ifndef NDEBUG
    assert(gpu_ad_it != gpu_ad_ptrs_.end());
#endif
    tmp_tensor->replace_gpu_ptr_without_free(gpu_ad_it->second);
  }
}

INSTANTIATE_CLASS(MemReducer);

}  // namespace ebird