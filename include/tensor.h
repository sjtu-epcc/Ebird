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
#include <cuda_header.h>
// #include <cuda_malloc.h>
#include <cudnn.h>
#include <cufft.h>
#include <initializer.h>
#include <math.h>
#include <stdio.h>
#include <stream_singleton.h>
#include <switch.h>
#include <sys/time.h>
#include <util/common.h>
#include <util/error_util.h>
// #include <util/lru.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <random>
#include <vector>

namespace ebird {

typedef enum TENSOR_TYPE {
  DATA = 0,
  // GRAD = 1,
  PARAM = 1,
  AUX = 2,
  BN_MEAN_VAR = 3,
  CONV_BUFF = 4,
  DATA_SOURCE = 5
} TENSOR_TYPE;

template <class value_type>
class Tensor {
private:
  TENSOR_TYPE data_type_;
  value_type *gpu_ptr_ = nullptr;  // gpu and cpu data are mutually exclusive
  value_type *cpu_ptr_ = nullptr;

  size_t GPU_id_;  // this identifies the GPU RAM
  int layer_id_;   // this identifies the affiliated layer
  int network_id_;
  size_t N_;
  size_t C_;
  size_t H_;
  size_t W_;

  /*---tensor_id----*/
  static size_t tensor_counter_;
  const size_t tensor_id_ = 0;

  // CUDNN configuration
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t cudnn_tensor_format_;  // default with NCHW
  cudnnTensorDescriptor_t cudnn_tensor_desc_;

  void freeSpaceCPU() {
    if (cpu_ptr_ == nullptr) {
      return;
    }
    checkCudaErrors(cudaFreeHost(this->cpu_ptr_));
    this->cpu_ptr_ = nullptr;
  }

  void acquireSpaceGPU(long total);
  void freeSpaceGPU(MemMode target = CPU);

public:
  int hit_cnt = 0, miss_cnt = 0, into_cnt = 0;

  Tensor(size_t n, size_t c, size_t h, size_t w,
         std::vector<Tensor<value_type> *> *reg, TENSOR_TYPE dtype,
         int layer_id)
      : tensor_id_(tensor_counter_) {
    assert(n >= 1);
    assert(c >= 1);
    assert(h >= 1);
    assert(w >= 1);

    this->GPU_id_ = 0;

    // #ifdef TENSOR_REUSE
    //         gpu_mem_blocks = GpuMemBlocks<value_type>::getInstance();
    // #endif

    this->data_type_ = dtype;
    this->layer_id_ = layer_id;

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

    this->cudnn_tensor_format_ = CUDNN_TENSOR_NCHW;
    checkCUDNN(cudnnCreateTensorDescriptor(&cudnn_tensor_desc_));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc_,
                                          this->cudnn_tensor_format_,
                                          this->cudnn_data_type_, n, c, h, w));
    const size_t total_size = n * c * h * w;

    if (this->data_type_ == PARAM || this->data_type_ == AUX ||
        this->data_type_ == BN_MEAN_VAR) {
      acquireSpaceCPU(n * c * h * w);
      acquireSpaceGPU(n * c * h * w);
    }
    if (this->data_type_ == DATA_SOURCE) {
    acquireSpaceGPU(n * c * h * w);
    }

    this->N_ = n;
    this->C_ = c;
    this->H_ = h;
    this->W_ = w;

    reg->push_back(this);
    /*---init-counter---*/
    tensor_counter_++;

#ifndef NDEBUG
    const size_t total_size_bytes = sizeof(value_type) * n * c * h * w;

    if (this->data_type_ == DATA) {
      printf("create tensor:%p DATA gpu_ptr:%p size: %zu byte\n", this,
             gpu_ptr_, total_size_bytes);
    } else if (this->data_type_ == PARAM) {
      printf("create tensor:%p PARAM gpu_ptr:%p size: %zu byte\n", this,
             gpu_ptr_, total_size_bytes);
    } else if (this->data_type_ == AUX) {
      printf("create tensor:%p AUX gpu_ptr:%p size: %zu byte\n", this, gpu_ptr_,
             total_size_bytes);
    } else if (this->data_type_ == BN_MEAN_VAR) {
      printf("create tensor:%p BN_MEAN_VAR gpu_ptr:%p size: %zu byte\n", this,
             gpu_ptr_, total_size_bytes);
    } else if (this->data_type_ == CONV_BUFF) {
      printf("create tensor:%p CONV_BUFF gpu_ptr:%p size: %zu byte\n", this,
             gpu_ptr_, total_size_bytes);
    } else if (this->data_type_ == DATA_SOURCE) {
      printf("create tensor:%p DATA_SOURCE gpu_ptr:%p size: %zu byte\n", this,
             gpu_ptr_, total_size_bytes);
    } else {
      printf("unsupported type@%d tensor.h line 86\n", this->data_type_);
      exit(1);
    }
#endif
  }

  ~Tensor() {
    if (cpu_ptr_ != nullptr) cudaFreeHost(cpu_ptr_);
    if (gpu_ptr_ != nullptr) gpu_ptr_ = nullptr;
    checkCUDNN(cudnnDestroyTensorDescriptor(cudnn_tensor_desc_));
  }

  /*----utility functions----*/

  /**
   * NCHW, layer_id, data_type, data
   */
  void gen_description(char *buff, size_t *len_in_byte) {
    value_type _n = N_, _c = C_, _h = H_, _w = W_;
    value_type _layer_id = layer_id_, _type = data_type_;

    size_t SIZE = sizeof(value_type);
    memcpy(buff, &_n, SIZE);
    memcpy(buff + 1 * SIZE, &_c, SIZE);
    memcpy(buff + 2 * SIZE, &_h, SIZE);
    memcpy(buff + 3 * SIZE, &_w, SIZE);
    memcpy(buff + 4 * SIZE, &_layer_id, SIZE);
    memcpy(buff + 5 * SIZE, &_type, SIZE);

    this->GPUtoCPU();

    memcpy(buff + 6 * SIZE, this->cpu_ptr_, N_ * C_ * H_ * W_ * SIZE);

    *len_in_byte = (6 + N_ * C_ * H_ * W_) * SIZE;
  }

  inline size_t get_N() { return this->N_; }

  inline size_t get_C() { return this->C_; }

  inline size_t get_H() { return this->H_; }

  inline size_t get_W() { return this->W_; }

  inline size_t get_scalar_count() {
    return this->get_N() * this->get_C() * this->get_H() * this->get_W();
  }

  inline size_t get_mem_size() {
    const size_t total_size_bytes =
        sizeof(value_type) * this->N_ * this->C_ * this->H_ * this->W_;
    return total_size_bytes;
  }

  void acquireSpaceCPU(long total) {
    assert(cpu_ptr_ == nullptr);
    assert(total > 0);
    checkCudaErrors(
        cudaMallocHost(&(this->cpu_ptr_), total * sizeof(value_type)));
  }

  void reshape(size_t n, size_t c, size_t h, size_t w) {
    assert(N_ * C_ * H_ * W_ == n * c * h * w);
    this->N_ = n;
    this->C_ = c;
    this->H_ = h;
    this->W_ = w;
    checkCUDNN(cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc_,
                                          this->cudnn_tensor_format_,
                                          this->cudnn_data_type_, n, c, h, w));
  }

  void replace_data(value_type *new_cpu_ptr, value_type *new_gpu_ptr = nullptr);

  inline void replace_gpu_ptr_without_free(value_type *&new_gpu_ptr) {
    // DLOG(INFO) << "---FILE_NAME: " << __FILE__ << "---LINE: " << __LINE__
    //            << "---";
    this->gpu_ptr_ = new_gpu_ptr;
  }
  inline void replace_gpu_ptr(value_type *&new_gpu_ptr) {
    checkCudaErrors(cudaFreeHost(this->gpu_ptr_));
    this->gpu_ptr_ = new_gpu_ptr;
  }

  inline int get_tensor_id() { return this->tensor_id_; }
  inline int get_layer_id() { return this->layer_id_; }

  inline TENSOR_TYPE get_type() { return this->data_type_; }

  inline value_type *get_cpu_ptr() { return this->cpu_ptr_; }

  inline value_type *get_gpu_ptr() { return this->gpu_ptr_; }

  inline cudnnTensorDescriptor_t get_tensor_desc() {
    return this->cudnn_tensor_desc_;
  }

  inline cudnnTensorFormat_t get_tensor_format() {
    return this->cudnn_tensor_format_;
  }

  void GPUtoCPU();

  void CPUtoGPU();

  void sync_cpu_to_gpu();

  void sync_gpu_to_cpu();

  void init(Initializer<value_type> *initializer);

  void printTensorNoDebug(const char *str);

  void printTensor(const char *str);

  void printTensorFirst(const char *str);

  void writeToFile(const char *str);

  void hostRegister();

  void resizeTensor(size_t n, size_t c, size_t h, size_t w);

  void copy(Tensor<value_type> *t, cudaStream_t &cur_stream,
            int src_start_idx = -1, int src_end_idx = -1,
            int dst_start_idx = -1, int dst_end_idx = -1);

  value_type get_scalar(const size_t n, const size_t c, const size_t h,
                        const size_t w);

  void set_scalar(const size_t n, const size_t c, const size_t h,
                  const size_t w, const value_type t);

  /*---math functions-------*/
  void scale(value_type s, cudaStream_t &cur_stream);

  void sum(Tensor<value_type> *t, cudaStream_t &cur_stream);

  value_type squared_sum(cublasHandle_t *handle);
};

}  // namespace ebird
