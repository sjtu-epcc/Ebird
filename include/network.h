/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:11:21 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once

#include <registry.h>

#include <deque>
#include <thread>
// #include <mem_control.h>
// #include <cuda_malloc.h>
#include <layer/base_layer.h>
#include <layer/cudnn_convolution_layer.h>
#include <layer/data_layer.h>
#include <mem_reducer.h>
#include <stream_singleton.h>
#include <util/common.h>
#include <util/error_util.h>
#include <util/mem_util.h>

#include <string>

namespace ebird {

template <class value_type>
class Network {
 private:
  /*-network configurations-*/
  size_t GPU_id_;
  // TODO set batch_szie according to the first data layer
  size_t batch_size_;
  /* network_id*/

  static size_t network_cnt_;
  const size_t network_id_ = 0;
  static size_t network_active_;

  bool is_forward_setup_;
  bool is_infer_ready_;
  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudaStream_t stream_;

  MemReducer<value_type> mem_reducer_;

  std::vector<BaseLayer<value_type> *> net_infer_layers_;

  BaseLayer<value_type> *infer_data_layer_ = nullptr;

  Registry<value_type> *registry_;

  void createHandles() {
    checkCUDNN(cudnnCreate(&cudnn_handle_));
    cudnnSetStream(cudnn_handle_, stream_);
    checkCublasErrors(cublasCreate(&cublas_handle_));
    cublasSetStream(cublas_handle_, stream_);
  }

  void destroyHandles() {
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    checkCublasErrors(cublasDestroy(cublas_handle_));
  }

  void forward_infer(NetworkStage stage);

  void forward_kernel(NetworkStage stage, BaseLayer<value_type> *b,
                      std::vector<value_type> *loss);

  void fsetup_kernel(BaseLayer<value_type> *start_layer);

  void write_tensor(int32_t layer_id, Tensor<value_type> *t,
                    std::ofstream *out);
  void read_tensor(int32_t *layer_id, Tensor<value_type> **t,
                   std::ifstream *in);

  void meta_setup() {
    // CAUTION: the sequence of registry and mem_controller matters
    this->batch_size_ = registry_->get_batch_size();
#ifndef NDEBUG
    registry_->print_net_infer_route();
    registry_->print_net_layers();
#endif
    registry_->register_tensors_by_layers();

#ifndef NDEBUG
    registry_->print_tensors_by_layers();

    printf("*****************network dependencies***************\n");
    printf("---------------------FORWARD-------------------------\n");
    registry_->print_forward_dependency();
    printf("****************************************************\n");
#endif
  }

  void mem_setup() {
    mem_reducer_.init(this->registry_);
    mem_reducer_.scanNet();
    mem_reducer_.gpuAlloc();
  }

  void inference_setup();

 public:
  Network() : network_id_(network_cnt_) {
    if (network_id_ == 1) {
      google::InitGoogleLogging("");
      FLAGS_logtostderr = 1;
      FLAGS_log_dir = "./";
    }
    GPU_id_ = 0;
    // TODO set gpu_id, bind to taget GPU
    network_cnt_++;
    network_active_++;

    stream_ = StreamSingleton::get_compute_stream(network_id_);

    registry_ = new Registry<value_type>();

    // #ifdef TENSOR_REUSE
    //         printf("Using Tensor_reuse !!!!\n");
    // #endif
    // set affinity
    // TODO set cpu affinity
    // set_main_thread_cpu_affinity(1);

    // init_gpu_mem_query();

    is_forward_setup_ = false;
    is_infer_ready_ = false;

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
    createHandles();
  };

  ~Network() {
    // we first destroy data layer because the ParallelReader must be
    // destroyed before registry
    delete infer_data_layer_;

    delete registry_;

    // query_stop = true;
    // query_thread->join();
    // the sequence matters
    destroyHandles();
    // all tensors will be deleted in the registry class

    // finish all computation, destroy compute stream
    network_active_--;
    if (network_active_ == 0) StreamSingleton::destory_stream();

    // destroy global blasx_malloc_t
    // blasx_gpu_singleton::destroy_all_instance();
  }

  cudnnHandle_t *get_cudnn_handle() { return &(this->cudnn_handle_); }

  cublasHandle_t *get_cublas_handle() { return &(this->cublas_handle_); }

  Registry<value_type> *get_registry() { return this->registry_; }

  void fsetup(BaseLayer<value_type> *start_layer) {
    if (this->infer_data_layer_ == nullptr) {
      this->infer_data_layer_ = start_layer;
    } else {
      printf(
          "forward setup inference data layer could only be set once!! "
          "line 12@network.cc\n");
      exit(1);
    }
    // batch_size = start_
    fsetup_kernel(start_layer);
    this->is_forward_setup_ = true;

    meta_setup();
    mem_setup();
    inference_setup();
  }

  // void setup_infer(BaseLayer<value_type> *infer_data_layer,
  //                 size_t iter);

  // value_type forward(NetworkStage stage) {
  //     assert(this->train_data_layer != nullptr);
  //     BaseLayer<value_type> *n = this->train_data_layer;
  //     std::vector<value_type> loss;
  //     forward_kernel(stage, n, &loss);
  //     return loss[0];
  // }
  void inference(value_type *input_gpu_ptr = nullptr);
  size_t get_output_size() { return this->registry_->get_output_size(); }
  cudaStream_t get_stream() { return this->stream_; }
};

}  // namespace ebird
