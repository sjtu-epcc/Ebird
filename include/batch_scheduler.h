/*
 * Created Date: Tuesday, June 11th 2019, 4:31:12 pm
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

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "network.h"
#include "stream_singleton.h"
#include "tensor.h"
#include "util/common.h"
#include "util/error_util.h"

namespace ebird {

class Throttle {
 private:
  int concurrency_;
  int active_;
  std::mutex mutex_;
  std::condition_variable throttle_cond_;

 public:
  explicit Throttle(int concurrency) : concurrency_(concurrency), active_(0) {}
  void inc();
  void dec();
  void inc_cur(int cur);
  void dec_cur(int cur);
};
template <class value_type>
class WorkerConf {
 public:
  std::condition_variable cond_;
  std::mutex mutex_;

  size_t batchsize_;
  int start_index_;

  bool input_ready_;
  value_type *input_dev_;
  WorkerConf(size_t batchsize)
      : batchsize_(batchsize),
        start_index_(0),
        input_ready_(false),
        input_dev_(nullptr) {}
};

template <class value_type>
class WorkerPool {
 public:
  size_t input_size_;
  size_t output_size_;

  size_t max_requests_;
  size_t active_requests_;

  std::mutex idle_mutex_;
  std::map<size_t, std::deque<void *>, std::greater<size_t>> idle_worker_;
  // std::unordered_map<size_t, std::deque<void *>> busy_worker_;
  std::unordered_map<void *, std::unique_ptr<WorkerConf<value_type>>>
      workers_conf_;
  // std::unordered_map<void *, std::unique_ptr<std::condition_variable>>
  //     worker_cond_;
  // std::unordered_map<void *, std::unique_ptr<std::mutex>> worker_mutex_;
  // std::unordered_map<void *, bool> input_ready_;
  // std::unordered_map<void *, size_t> worker_bachszie_;
  // TODO how to use the setup the input_gpu_ptr before inference

  // std::unordered_map<void *, void **> worker_input_gpu_ptr_;

  WorkerPool(size_t input_size, size_t output_size)
      : input_size_(input_size),
        output_size_(output_size),
        max_requests_(0),
        active_requests_(0) {}

  void add_worker(void *network_ptr, size_t batch_size);
  size_t schedule_worker_by_size(int start_index, size_t &waiting_size,
                                 value_type *waiting_addr);
};

template <class value_type>
class MemPool {
 public:
  // bool stop_;
  size_t pool_size_;

  size_t input_size_;
  size_t output_size_;

  value_type *gpu_mem_pool;  // initial ptr of GPU resident memory pool
  // value_type *output_gpu_ptr_;
  value_type **output_cpu_ptr_;  // cpu ptr array to store the output ptr, the
                                 // same size as gpu memory pool

  // input sync
  std::mutex input_mutex_;
  std::condition_variable input_cond_;
  int capacity_;
  int input_index_;
  int remain_;
  int process_index_;
  int comp_index_;
  std::vector<bool> completed_;
  std::mutex comp_mutex_;
  // output sync
  std::vector<std::unique_ptr<std::mutex>> output_mutex_;
  std::vector<std::unique_ptr<std::condition_variable>> output_cond_;
  std::vector<bool> output_ready_;

  cudaStream_t input_stream_;
  cudaStream_t output_stream_;

  cudaEvent_t *input_event_;

  MemPool(size_t pool_size, size_t input_size, size_t output_size)
      : pool_size_(pool_size),
        input_size_(input_size),
        output_size_(output_size) {
    /**
     * @brief
     * @note
     * @param  pool_size:
     * @param  input_size:  dimension of image, excluding the size of type
     * @param  output_size:
     * @retval
     */
    checkCudaErrors(cudaMalloc(&gpu_mem_pool,
                               pool_size_ * input_size_ * sizeof(value_type)));

    capacity_ = pool_size_;
    input_index_ = 0;
    process_index_ = 0;
    comp_index_ = 0;

    output_cpu_ptr_ = new value_type *[pool_size_];

    input_stream_ = StreamSingleton::get_input_stream();
    output_stream_ = StreamSingleton::get_output_stream();

    input_event_ = new cudaEvent_t[pool_size_];

    for (size_t i = 0; i < pool_size_; i++) {
      checkCudaErrors(cudaEventCreate(&input_event_[i]));
      output_mutex_.emplace_back(new std::mutex());
      output_cond_.emplace_back(new std::condition_variable());
      output_ready_.emplace_back(false);
      completed_.emplace_back(true);
    }
    completed_[0] = false;
  }
  std::tuple<int, size_t, value_type *> waiting_schedule() {
    size_t waiting_size;
    {
      std::lock_guard<std::mutex> lock_input(input_mutex_);
      waiting_size = input_index_ >= process_index_
                         ? input_index_ - process_index_
                         : pool_size_ - process_index_;
    }
    value_type *waiting_addr = gpu_mem_pool + process_index_ * input_size_;

    std::tuple<int, size_t, value_type *> ret{process_index_, waiting_size,
                                              waiting_addr};
    return ret;
  }
  void update_process(size_t processed) {
    std::lock_guard<std::mutex> lock_comp(comp_mutex_);
    completed_[process_index_] = false;
    DLOG(INFO) << "Input Index: " << input_index_
               << " Process Index:" << process_index_
               << " Complete Index: " << comp_index_;
    process_index_ = (process_index_ + processed) % pool_size_;
  }
  // no read write protect
  void update_capacity() {
    // DLOG(INFO) << completed_[comp_index_];
    DLOG(INFO) << "completed true index: " << comp_index_;
    while (completed_[comp_index_] == true && comp_index_ != process_index_) {
      {
        DLOG(INFO) << "completed true index: " << comp_index_;
        std::lock_guard<std::mutex> lock_input(input_mutex_);
        comp_index_ = (comp_index_ + 1) % pool_size_;
        capacity_++;
      }
      input_cond_.notify_all();
    }
  }
};

template <class value_type>
class BatchScheduler {
 private:
  // stop
  bool stop_;
  std::mutex stop_mutex_;
  // test threads
  std::vector<std::thread> test_threads_;  // threads for testing

  // worker threads
  std::vector<std::thread> worker_threads_;

  // scheduler thread

  std::thread scheduler_thread_;

  // GPU memory pool sync
  std::shared_ptr<MemPool<value_type>> mem_pool_;

  // worker sync
  std::shared_ptr<WorkerPool<value_type>> worker_pool_;

 public:
  BatchScheduler(size_t pool_size, size_t input_size, size_t output_size)
      : stop_(false) {
    // checkCudaErrors(cudaMalloc(&output_gpu_ptr_, pool_size_ *
    // output_size_));
    mem_pool_ = std::make_shared<MemPool<value_type>>(pool_size, input_size,
                                                      output_size);
    worker_pool_ =
        std::make_shared<WorkerPool<value_type>>(input_size, output_size);
  }

  void add_worker(Network<value_type> *network_ptr, size_t batch_size);

  void start_worker();

  void start_test(size_t num_tests, int concurrency = 10, bool bursty = false,
                  int inc_bursty = 0);

  void send_req(value_type *input_cpu_ptr, value_type *output_cpu_ptr,
                std::shared_ptr<MemPool<value_type>> mem_pool);

  void schedule(size_t max_requests);

  void run_net() {
    Network<value_type> *tmp_network =
        (Network<value_type> *)worker_pool_->idle_worker_.begin()
            ->second.front();
    std::thread thread_1 = std::thread([=] { tmp_network->inference(); });
    thread_1.join();
  }
};

}  // namespace ebird
