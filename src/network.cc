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
#include <network.h>
#include <tensor.h>
#include <fstream>

namespace ebird {

template <class value_type>
size_t Network<value_type>::network_cnt_ = 1;

template <class value_type>
size_t Network<value_type>::network_active_ = 0;

template <class value_type>
void Network<value_type>::inference_setup() {
  std::vector<std::pair<int, NetDirect>> net_infer_route =
      registry_->get_net_infer_route();
  std::map<int, void *> net_layers = registry_->get_net_layers();

  for (size_t i = 0; i < net_infer_route.size(); i++) {
    int layer_id = net_infer_route[i].first;
    BaseLayer<value_type> *b =
        (BaseLayer<value_type> *)net_layers.find(layer_id)->second;
    net_infer_layers_.emplace_back(b);
    DLOG(INFO) << "layer id: " << b->get_base_id()
               << "total layer : " << b->get_layer_num();
  }
}

template <class value_type>
void Network<value_type>::fsetup_kernel(BaseLayer<value_type> *b) {
  if (b == nullptr) return;

  // conduct forward computation
  b->fcounter_inc();

  if (b->get_fcounter() < b->get_prev_size()) {
    return;
  }
  b->forward_setup(registry_, &cudnn_handle_);
  this->registry_->register_net_layers(b->get_base_id(), (void *)b);
  this->registry_->register_net_infer_route(b->get_base_id(), FORWARD);

  std::vector<BaseLayer<value_type> *> next = b->get_next();
  if (next.size() == 1) {
    // regular network layer
    fsetup_kernel(next[0]);
  } else if (next.size() > 1) {
    // fork layer
    for (size_t i = 1; i < next.size(); i++) {
      fsetup_kernel(next[i]);
    }
    fsetup_kernel(next[0]);
  }
  b->reset_fc_counter();
}

// TODO optimize the infer progress
template <class value_type>
void Network<value_type>::forward_infer(NetworkStage stage) {
  for (auto layer : net_infer_layers_) {
    layer->forward(stage, &cublas_handle_, &cudnn_handle_, registry_);
#ifndef NDEBUG
    checkCudaErrors(cudaStreamSynchronize(layer->_get_stream_()));
#endif
  }
  // std::vector<std::pair<int, NetDirect>> net_infer_route =
  //     registry_->get_net_infer_route();
  // std::map<int, void *> net_layers = registry_->get_net_layers();

  // for (size_t i = 0; i < net_infer_route.size(); i++) {
  //   // if (net_infer_route[i].second == FORWARD) {
  //   int layer_id = net_infer_route[i].first;

  //   BaseLayer<value_type> *b =
  //       (BaseLayer<value_type> *)net_layers.find(layer_id)->second;
  //   b->forward(stage, &cublas_handle_, &cudnn_handle_, registry_);
  // }
}
// namespace ebird

template <class value_type>
void Network<value_type>::inference(value_type *input_gpu_ptr) {
  // assert(this->infer_data_layer != nullptr);
  // assert(this->train_data_layer != nullptr);
  // let's swap the head of data layer
  // please note prev matters!!!
  // the first network layer will need prev to figure out the input
  // BaseLayer<value_type> *train_l = this->train_data_layer;
  // BaseLayer<value_type> *test_l = this->infer_data_layer;
  // std::vector<BaseLayer<value_type> *> next_ls = train_l->get_next();
  // assert(next_ls.size() == 1);
  // BaseLayer<value_type> *next_l = next_ls[0];
  // next_l->switch_prev_l_to(test_l);

  // BaseLayer<value_type> *start = this->infer_data_layer;

  // checkCudaErrors(cudaDeviceSynchronize());
  // std::vector<value_type> tmp;
  // DLOG(INFO) << "---------------------FILE_NAME: " << __FILE__
  //            << "-------------LINE: " << __LINE__ << "----------";
  DLOG(INFO) << __FILE__ << " " << __LINE__;
  if (input_gpu_ptr != nullptr) this->registry_->push_input(input_gpu_ptr);
  // DLOG(INFO) << "---------------------FILE_NAME: " << __FILE__
  //            << "-------------LINE: " << __LINE__ << "----------";
  DLOG(INFO) << __FILE__ << " " << __LINE__;
  forward_infer(NET_INFER);
  DLOG(INFO) << __FILE__ << " " << __LINE__;

  // DLOG(INFO) << "---------------------FILE_NAME: " << __FILE__
  //            << "-------------LINE: " << __LINE__ << "----------";
  // printf("-------test accuracy--top 1 %f top 5 %f-------\n",
  // infer_accuracy_top1, infer_accuracy_top5);
}

// template <class value_type>
// std::vector<double> Network<value_type>::network_perf_profile()
// {

//     std::vector<std::pair<int, NetDirect>> net_infer_route =
//     reg->get_net_infer_route(); std::map<int, void *> net_layers =
//     reg->get_net_layers(); double max_mem_used = 0; for (size_t i = 0; i <
//     net_infer_route.size(); i++)
//     {
//         int layer_id = net_infer_route[i].first;
//         NetDirect dir = net_infer_route[i].second;
//         // stash tensors
//         std::pair<double, double> stat =
//         mem_controller.stash_tensors_for_profile(layer_id, dir); double
//         mem_stage_time = stat.first; double total_mem = stat.second; double
//         curt_mem = BYTE_TO_MB(query_used_mem()); if (curt_mem > max_mem_used)
//             max_mem_used = curt_mem;
//         // execution
//         BaseLayer<value_type> *b = (BaseLayer<value_type>
//         *)net_layers.find(layer_id)->second; double start = get_cur_time();
//         for (size_t j = 0; j < 10; j++)
//         {
//             if (dir == FORWARD)
//             {
//                 b->forward(NET_TRAIN, &cublas_handle, &cudnn_handle, reg);
//             }
//             else if (dir == BACKWARD)
//             {
//                 NO_TRAIN;
//             }
//             //cudaStreamSynchronize(stream);
//         }
//         LAYER layer_type = b->get_layer_type();
//         double end = get_cur_time();
//         double avg_time = (end - start) / 10.0f;
//         // mem_controller.free_tensors_for_profile(layer_id, dir);
//         double mem_time = mem_stage_time;
//         printf("at layer id:%3d, type:%3d, compute_time:%5.5f,
//         memory_time:%5.5f, total_mem:%5.5f\n", layer_id, layer_type,
//         avg_time, mem_time, total_mem);
//     }
//     printf("Max Memory used in profile:%f\n", max_mem_used);
//     return std::vector<double>();
// }

INSTANTIATE_CLASS(Network);

}  // namespace ebird
