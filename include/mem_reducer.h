/*
 * Created Date: Thursday, February 6th 2020, 3:24:12 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:32:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2020 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once

#include <layer/base_layer.h>
#include <tensor.h>
#include <unordered_map>
#include <vector>

namespace ebird {

template <class value_type>
class MemReducer {
 private:
  // allocation
  // busy tensor key: tensor; value: gpu address id
  std::unordered_map<Tensor<value_type> *, int> busy_tensors_;
  // key: size; value: gptr id
  std::multimap<int, int> idle_gpu_ads_;
  // key:tensor; value: gpu address id
  std::unordered_map<Tensor<value_type> *, int> tensor_gpu_ads_;
  // key: gpu address id; value: size
  std::unordered_map<int, int> gpu_ad_sizes_;
  // key: gpu address id; value: ptr
  std::unordered_map<int, value_type *> gpu_ad_ptrs_;

  // topology of the network
  Registry<value_type> *registry_;
  std::unordered_map<int, std::vector<Tensor<value_type> *>>
      regulated_tensors_by_layer_;

  int max_layer_cnt_;
  std::unordered_map<int, std::vector<int>> subsequent_forward_;

  // build topology from registry
  std::vector<int> &get_subsequent_layers(int cur_layer_id);
  bool is_used_by_layer(int layer_id, Tensor<value_type> *t);
  bool is_freeable_afterwards(int cur_layer_id, Tensor<value_type> *t);
  void set_regulated_tensors();

 public:
  MemReducer() {}
  ~MemReducer() {}

  void init(Registry<value_type> *reg);

  void scanNet();
  void gpuAlloc();
};
}  // namespace ebird