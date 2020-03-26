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
#include <registry.h>

namespace ebird {

template <class value_type>
void Registry<value_type>::print_tensors_by_layers() {
  typename std::map<int, std::vector<Tensor<value_type> *>>::iterator it =
      tensor_by_layer_.begin();
  int data_c = 0;
  int conv_buff_c = 0;
  long total_data = 0;
  long total_grad = 0;
  long total_aux = 0;
  long total_param = 0;
  long total_conv_buff = 0;
  long total_bn_mean_var = 0;
  long total_data_source = 0;

  for (it = tensor_by_layer_.begin(); it != tensor_by_layer_.end(); ++it) {
    int layer_id = it->first;
    std::vector<Tensor<value_type> *> tensors = it->second;
    for (size_t i = 0; i < tensors.size(); i++) {
      TENSOR_TYPE type = tensors[i]->get_type();
      if (type == DATA) {
        data_c += 1;
        total_data += ((long)tensors[i]->get_scalar_count());
        printf("@layer:%d->tensor:%p DATA  size %.3fMB\n", layer_id, tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      } else if (type == PARAM) {
        total_param += ((long)tensors[i]->get_scalar_count());
        printf("@layer:%d->tensor:%p PARAM  size %.3fMB\n", layer_id,
               tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      } else if (type == AUX) {
        total_aux += ((long)tensors[i]->get_scalar_count());
        printf("@layer:%d->tensor:%p AUX  size %.3fMB\n", layer_id, tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      } else if (type == BN_MEAN_VAR) {
        total_bn_mean_var += ((long)tensors[i]->get_scalar_count());
        printf("@layer:%d->tensor:%p BN_MEAN_VAR  size %.3fMB\n", layer_id,
               tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      } else if (type == CONV_BUFF) {
        conv_buff_c += 1;
        total_conv_buff += ((long)tensors[i]->get_scalar_count());
        printf("@layer:%d->tensor:%p CONV_BUFF  size %.3fMB\n", layer_id,
               tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      } else if (type == DATA_SOURCE) {
        total_data_source += ((long)tensors[i]->get_scalar_count());
        printf("@layer:%d->tensor:%p DATA_SOURCE  size %.3fMB\n", layer_id,
               tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      } else {
        printf("@layer:%d->tensor:%p unspecified  size %.3fMB\n", layer_id,
               tensors[i],
               (double)tensors[i]->get_mem_size() / 1024.0f / 1024.0f);
      }
    }
  }

  double total_aux_mem = total_aux * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_data_mem = total_data * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_grad_mem = total_grad * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_param_mem = total_param * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_conv_buff_mem =
      total_conv_buff * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_bn_mean_var_mem =
      total_bn_mean_var * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_data_source_mem =
      total_data_source * sizeof(value_type) / 1024.0f / 1024.0f;
  double total_mem = total_data_mem + total_conv_buff_mem + total_param_mem +
                     total_grad_mem + total_aux_mem + total_bn_mean_var_mem +
                     total_data_source_mem;

  printf("TOTAL %d DATA: %ld->%fMB\n", data_c, total_data, total_data_mem);
  printf("TOTAL %d CONV BUFF: %ld->%fMB \n", conv_buff_c, total_conv_buff,
         total_conv_buff_mem);
  printf("TOTAL PARAMS: %ld->%fMB \n", total_param, total_param_mem);
  printf("TOTAL GRAD: %ld->%fMB \n", total_grad, total_grad_mem);
  printf("TOTAL AUX: %ld->%fMB \n", total_aux, total_aux_mem);
  printf("TOTAL BN_MEAN_VAR: %ld->%fMB \n", total_bn_mean_var,
         total_bn_mean_var_mem);
  printf("TOTAL DATA_SOURCE: %ld->%fMB \n", total_data_source,
         total_data_source_mem);
  printf("TOTAL MEM:%f\n", total_mem);
}

template <class value_type>
void Registry<value_type>::register_tensors_by_layers() {
  std::vector<Tensor<value_type> *> *all_tensors = this->get_vector();

  for (size_t i = 0; i < all_tensors->size(); i++) {
    Tensor<value_type> *t = (*all_tensors)[i];
    int layer_id = t->get_layer_id();
    typename std::map<int, std::vector<Tensor<value_type> *>>::iterator it =
        tensor_by_layer_.find(layer_id);
    if (it != tensor_by_layer_.end()) {
      it->second.push_back(t);
    } else {
      std::vector<Tensor<value_type> *> v;
      v.push_back(t);
      tensor_by_layer_.insert(
          std::pair<int, std::vector<Tensor<value_type> *>>(layer_id, v));
    }
  }
}

template <class value_type>
void Registry<value_type>::register_layer_param(
    int layer_id, Tensor<value_type> *t, std::map<int, Tensor<value_type> *> &m,
    const char *str) {
  typename std::map<int, Tensor<value_type> *>::iterator it = m.find(layer_id);
  if (it != m.end()) {
    printf("layer:%d %s already registered!! do nothing\n", layer_id, str);
    exit(1);
  } else {
    m.insert(std::make_pair(layer_id, t));
  }
#ifndef NDEBUG
  print_registry(m, str);
#endif
}

template <class value_type>
void Registry<value_type>::register_output(int source_layer_id,
                                           int dest_layer_id,
                                           Tensor<value_type> *t) {
  assert(t->get_type() == DATA || t->get_type() == DATA_SOURCE);
  LayerKey k(source_layer_id, dest_layer_id);
  typename std::map<LayerKey, Tensor<value_type> *>::iterator it =
      outputs_.find(k);

  if (it != outputs_.end()) {
    printf(
        "source layer:%d dest layer:%d output tensor already "
        "registered!!\n",
        source_layer_id, dest_layer_id);
    exit(1);
  } else {
    outputs_.insert(std::make_pair(k, t));
  }

#ifndef NDEBUG
  print_registry(outputs_, "output");
#endif
}

template <class value_type>
Tensor<value_type> *Registry<value_type>::get_layer_param(
    int layer_id, std::map<int, Tensor<value_type> *> &m) {
  typename std::map<int, Tensor<value_type> *>::iterator it = m.find(layer_id);
  if (it != m.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

template <class value_type>
Tensor<value_type> *Registry<value_type>::get_reg_output(int source_layer_id,
                                                         int dest_layer_id) {
  LayerKey k(source_layer_id, dest_layer_id);
  typename std::map<LayerKey, Tensor<value_type> *>::iterator it =
      outputs_.find(k);
  if (it != outputs_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

template <class value_type>
void Registry<value_type>::register_forward_dependency(int layer_id,
                                                       Tensor<value_type> *t) {
  // to register the forward dependency by layers
  typename std::map<int, std::vector<Tensor<value_type> *>>::iterator it1 =
      forward_dependency_.find(layer_id);
  if (it1 != forward_dependency_.end()) {
    it1->second.push_back(t);
  } else {
    std::vector<Tensor<value_type> *> v;
    v.push_back(t);
    forward_dependency_.insert(
        std::pair<int, std::vector<Tensor<value_type> *>>(layer_id, v));
  }
  // to register the forward dependency by tensors
  typename std::map<Tensor<value_type> *, std::vector<int>>::iterator it2 =
      forward_dependency_by_tensor_.find(t);
  if (it2 != forward_dependency_by_tensor_.end()) {
    it2->second.push_back(layer_id);
  } else {
    std::vector<int> v;
    v.push_back(layer_id);
    forward_dependency_by_tensor_.insert(
        std::pair<Tensor<value_type> *, std::vector<int>>(t, v));
  }
}

template <class value_type>
bool Registry<value_type>::is_included(std::vector<Tensor<value_type> *> &v,
                                       Tensor<value_type> *t) {
  for (size_t i = 0; i < v.size(); i++) {
    if (v[i] == t) return true;
  }
  return false;
}

template <class value_type>
void Registry<value_type>::print_dependency_by_tensors(
    std::map<Tensor<value_type> *, std::vector<int>> &m, NetDirect dir) {
  typename std::map<Tensor<value_type> *, std::vector<int>>::iterator it =
      m.begin();
  for (it = m.begin(); it != m.end(); ++it) {
    std::vector<int> layers = it->second;
    for (size_t i = 0; i < layers.size(); i++) {
      TENSOR_TYPE type = it->first->get_type();
      if (dir == FORWARD) {
        if (type == DATA) {
          printf("@tensor:%p forward needed by layer:%d DATA\n", it->first,
                 layers[i]);
        } else if (type == PARAM) {
          printf("@tensor:%p forward needed by layer:%d PARAM\n", it->first,
                 layers[i]);
        } else if (type == AUX) {
          printf("@tensor:%p forward needed by layer:%d AUX\n", it->first,
                 layers[i]);
        } else if (type == BN_MEAN_VAR) {
          printf("@tensor:%p forward needed by layer:%d BN_MEAN_VAR\n",
                 it->first, layers[i]);
        } else if (type == CONV_BUFF) {
          printf("@tensor:%p forward needed by layer:%d CONV_BUFF\n", it->first,
                 layers[i]);
        } else if (type == DATA_SOURCE) {
          printf("@tensor:%p forward needed by layer:%d DATA_SOURCE\n",
                 it->first, layers[i]);
        } else {
          printf("@tensor:%p forward needed by layer:%d UNSPECIFIED\n",
                 it->first, layers[i]);
        }
      } else if (dir == BACKWARD) {
        if (type == DATA) {
          printf("@tensor:%p backward needed by layer:%d DATA\n", it->first,
                 layers[i]);
        } else if (type == PARAM) {
          printf("@tensor:%p backward needed by layer:%d PARAM\n", it->first,
                 layers[i]);
        } else if (type == AUX) {
          printf("@tensor:%p backward needed by layer:%d AUX\n", it->first,
                 layers[i]);
        } else if (type == BN_MEAN_VAR) {
          printf("@tensor:%p backward needed by layer:%d BN_MEAN_VAR\n",
                 it->first, layers[i]);
        } else if (type == CONV_BUFF) {
          printf("@tensor:%p backward needed by layer:%d CONV_BUFF\n",
                 it->first, layers[i]);
        } else if (type == DATA_SOURCE) {
          printf("@tensor:%p backward needed by layer:%d DATA_SOURCE\n",
                 it->first, layers[i]);
        } else {
          printf("@tensor:%p backward needed by layer:%d UNSPECIFIED\n",
                 it->first, layers[i]);
        }
      }
    }
  }
}

template <class value_type>
void Registry<value_type>::print_dependency(
    std::map<int, std::vector<Tensor<value_type> *>> &m, NetDirect dir) {
  long total_data = 0;
  long total_grad = 0;
  long total_aux = 0;
  long total_param = 0;
  long total_bn_mean_var = 0;
  long total_conv_buff = 0;
  long total_data_source = 0;

  std::vector<Tensor<value_type> *> dict;

  typename std::map<int, std::vector<Tensor<value_type> *>>::iterator it =
      m.begin();
  for (it = m.begin(); it != m.end(); ++it) {
    long layer_total = 0;
    int layer_id = it->first;
    std::vector<Tensor<value_type> *> tensors = it->second;

    for (size_t i = 0; i < tensors.size(); i++) {
      TENSOR_TYPE type = tensors[i]->get_type();
      bool is_old = is_included(dict, tensors[i]);
      if (!is_old) dict.push_back(tensors[i]);
      layer_total += ((long)tensors[i]->get_scalar_count());
      if (dir == FORWARD) {
        if (type == DATA) {
          if (!is_old) total_data += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d forward depends on tensor:%p DATA\n", layer_id,
                 tensors[i]);
        } else if (type == PARAM) {
          if (!is_old) total_param += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d forward depends on tensor:%p PARAM\n", layer_id,
                 tensors[i]);
        } else if (type == AUX) {
          if (!is_old) total_aux += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d forward depends on tensor:%p AUX\n", layer_id,
                 tensors[i]);
        } else if (type == BN_MEAN_VAR) {
          if (!is_old)
            total_bn_mean_var += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d forward depends on tensor:%p BN MEAN&VAR\n",
                 layer_id, tensors[i]);
        } else if (type == CONV_BUFF) {
          if (!is_old)
            total_conv_buff += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d forward depends on tensor:%p CONV_BUFF\n", layer_id,
                 tensors[i]);
        } else if (type == DATA_SOURCE) {
          total_data_source += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d->tensor:%p DATA_SOURCE\n", layer_id, tensors[i]);
        } else {
          printf("@layer:%d forward depends on tensor:%p UNSPECIFIED\n",
                 layer_id, tensors[i]);
        }
      } else if (dir == BACKWARD) {
        if (type == DATA) {
          if (!is_old) total_data += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d backward depends on tensor:%p DATA\n", layer_id,
                 tensors[i]);
        } else if (type == PARAM) {
          if (!is_old) total_param += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d backward depends on tensor:%p PARAM\n", layer_id,
                 tensors[i]);
        } else if (type == AUX) {
          if (!is_old) total_aux += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d backward depends on tensor:%p AUX\n", layer_id,
                 tensors[i]);
        } else if (type == BN_MEAN_VAR) {
          if (!is_old)
            total_bn_mean_var += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d backward depends on tensor:%p BN MEAN&VAR\n",
                 layer_id, tensors[i]);
        } else if (type == CONV_BUFF) {
          if (!is_old)
            total_conv_buff += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d backward depends on tensor:%p CONV_BUFF\n",
                 layer_id, tensors[i]);
        } else if (type == DATA_SOURCE) {
          total_data_source += ((long)tensors[i]->get_scalar_count());
          printf("@layer:%d->tensor:%p DATA_SOURCE\n", layer_id, tensors[i]);
        } else {
          printf("@layer:%d backward depends on tensor:%p UNSPECIFIED\n",
                 layer_id, tensors[i]);
        }
      }
    }
    double total_layer_mem = layer_total * sizeof(value_type) / 1000000.0f;
    printf("--------layer memory subtotal:%f--------\n", total_layer_mem);
  }
  double total_data_mem = total_data * sizeof(value_type) / 1000000.0f;
  double total_conv_buff_mem =
      total_conv_buff * sizeof(value_type) / 1000000.0f;
  double total_param_mem = total_param * sizeof(value_type) / 1000000.0f;
  double total_grad_mem = total_grad * sizeof(value_type) / 1000000.0f;
  double total_aux_mem = total_aux * sizeof(value_type) / 1000000.0f;
  double total_bn_mean_var_mem =
      total_bn_mean_var * sizeof(value_type) / 1000000.0f;
  double total_data_source_mem =
      total_data_source * sizeof(value_type) / 1000000.0f;
  double total_mem = total_data_mem + total_conv_buff_mem + total_param_mem +
                     total_grad_mem + total_aux_mem + total_bn_mean_var_mem +
                     total_data_source_mem;

  printf("TOTAL DATA: %ld->%fMB \n", total_data, total_data_mem);
  printf("TOTAL CONV BUFF: %ld->%fMB \n", total_conv_buff, total_conv_buff_mem);
  printf("TOTAL PARAMS: %ld->%fMB \n", total_param, total_param_mem);
  printf("TOTAL GRAD: %ld->%fMB \n", total_grad, total_grad_mem);
  printf("TOTAL AUX: %ld->%fMB \n", total_aux, total_aux_mem);
  printf("TOTAL BN_MEAN_VAR: %ld->%fMB \n", total_bn_mean_var,
         total_bn_mean_var_mem);
  printf("TOTAL DATA_SOURCE: %ld->%fMB \n", total_data_source,
         total_data_source_mem);
  printf("TOTAL MEM:%f\n", total_mem);
}

INSTANTIATE_CLASS(Registry);

}  // namespace ebird
