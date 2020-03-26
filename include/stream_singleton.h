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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <switch.h>
#include <mutex>

namespace ebird {

class StreamSingleton {
  cudaStream_t compute_stream_[MAX_STREAMS];
  cudaStream_t input_stream_;
  cudaStream_t output_stream_;
  static std::mutex mutex_;
  StreamSingleton() {
    for (auto i = 0; i < MAX_STREAMS; i++)
      cudaStreamCreate(&compute_stream_[i]);
    cudaStreamCreate(&input_stream_);
    cudaStreamCreate(&output_stream_);
  }

  // we must call the destructor explicitly
  ~StreamSingleton() {
    for (auto i = 0; i < MAX_STREAMS; i++)
      cudaStreamDestroy(compute_stream_[i]);
    cudaStreamDestroy(input_stream_);
    cudaStreamDestroy(output_stream_);
  }

  static StreamSingleton *instance_;

public:
  static cudaStream_t get_compute_stream(int stream_id) {
    if (instance_ == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (instance_ == nullptr) {
        instance_ = new StreamSingleton();
      }
    }
    return instance_->compute_stream_[stream_id];
  }

  static cudaStream_t get_input_stream() {
    if (instance_ == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (instance_ == nullptr) {
        instance_ = new StreamSingleton();
      }
    }
    return instance_->input_stream_;
  }

  static cudaStream_t get_output_stream() {
    if (instance_ == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (instance_ == nullptr) {
        instance_ = new StreamSingleton();
      }
    }
    return instance_->output_stream_;
  }
  // we must call the destructor explicitly
  static void destory_stream() { delete instance_; }
};

}  // namespace ebird
