/*
 * Created Date: Tuesday, June 11th 2019, 4:31:32 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Monday, March 16th 2020, 2:32:48 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include "batch_scheduler.h"

namespace ebird {

void Throttle::inc() {
  std::unique_lock<std::mutex> lock_cur(mutex_);
  while (active_ >= concurrency_) {
    throttle_cond_.wait(lock_cur);
  }
  active_++;
}
void Throttle::dec() {
  std::lock_guard<std::mutex> lock_cur(mutex_);
  active_--;
  throttle_cond_.notify_all();
}

void Throttle::inc_cur(int cur) {
  std::lock_guard<std::mutex> lock_cur(mutex_);
  concurrency_ += cur;
  throttle_cond_.notify_all();
}
void Throttle::dec_cur(int cur) {
  std::lock_guard<std::mutex> lock_cur(mutex_);
  concurrency_ -= cur;
}
template <class value_type>
void WorkerPool<value_type>::add_worker(void *network_ptr, size_t batch_size) {
  auto n_queue = idle_worker_.find(batch_size);
  if (n_queue != idle_worker_.end()) {
    n_queue->second.emplace_back(network_ptr);
  } else {
    idle_worker_.emplace(batch_size, std::deque<void *>{network_ptr});
  }
  std::unique_ptr<WorkerConf<value_type>> tmp_worker_conf(
      new WorkerConf<value_type>(batch_size));
  workers_conf_.emplace(network_ptr, std::move(tmp_worker_conf));
}

template <class value_type>
size_t WorkerPool<value_type>::schedule_worker_by_size(
    int start_index, size_t &waiting_size, value_type *waiting_addr) {
  size_t requests_cnt =
      std::min(max_requests_ - active_requests_, waiting_size);
  size_t ret_processed = 0;
  for (auto it_size = idle_worker_.begin(); it_size != idle_worker_.end();
       it_size++) {
    while (it_size->second.size() > 0 && it_size->first <= requests_cnt) {
      // activate the worker;
      void *worker;
      {
        std::unique_lock<std::mutex> lock_idle(idle_mutex_);

        active_requests_ += it_size->first;
        worker = it_size->second.front();
        it_size->second.pop_front();
      }
      {
        std::unique_lock<std::mutex> lock_input(workers_conf_[worker]->mutex_);
        workers_conf_[worker]->input_dev_ = waiting_addr;
        workers_conf_[worker]->start_index_ = start_index;
        workers_conf_[worker]->input_ready_ = true;
      }
      workers_conf_[worker]->cond_.notify_one();
      waiting_addr += input_size_ * it_size->first;
      requests_cnt -= it_size->first;
      ret_processed += it_size->first;
    }
  }
  return ret_processed;
}

template <class value_type>
void BatchScheduler<value_type>::add_worker(Network<value_type> *network_ptr,
                                            size_t batch_size) {
  worker_pool_->add_worker((void *)network_ptr, batch_size);
}

template <class value_type>
void BatchScheduler<value_type>::send_req(
    value_type *input_cpu_ptr, value_type *output_cpu_ptr,
    std::shared_ptr<MemPool<value_type>> mem_pool) {
  auto start = std::chrono::steady_clock::now();
  int cur_index = 0;
  void *cur_gpu_ptr;
  {
    std::unique_lock<std::mutex> lock_input(mem_pool->input_mutex_);
    while (mem_pool->capacity_ <= 0) {
      mem_pool->input_cond_.wait(lock_input);
    }
    cur_index = mem_pool->input_index_;

    mem_pool->input_index_ =
        (mem_pool->input_index_ + 1) % mem_pool->pool_size_;
    mem_pool->capacity_--;
    cur_gpu_ptr = mem_pool->gpu_mem_pool + mem_pool->input_size_ * cur_index;
    mem_pool->output_cpu_ptr_[cur_index] = output_cpu_ptr;

    start = std::chrono::steady_clock::now();
    // DLOG(INFO) << __FILE__ << " " << __LINE__;
    checkCudaErrors(cudaMemcpyAsync(
        cur_gpu_ptr, input_cpu_ptr, mem_pool->input_size_ * sizeof(value_type),
        cudaMemcpyHostToDevice, mem_pool->input_stream_));
    // DLOG(INFO) << __FILE__ << " " << __LINE__;
    checkCudaErrors(cudaEventRecord(mem_pool->input_event_[cur_index],
                                    mem_pool->input_stream_));
    // DLOG(INFO) << __FILE__ << " " << __LINE__;
  }
  {
    std::unique_lock<std::mutex> lock_output(
        *mem_pool->output_mutex_[cur_index]);
    while (!mem_pool->output_ready_[cur_index])
      mem_pool->output_cond_[cur_index]->wait(lock_output);
  }
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
  LOG(INFO) << "elapsed_time: " << elapsed_time;
  //   << " cur_index: " << cur_index;
}

template <class value_type>
void BatchScheduler<value_type>::start_test(size_t num_tests, int concurrency,
                                            bool bursty, int inc_bursty) {
  LOG(INFO) << num_tests << " tests remain, Using " << concurrency
            << "threads to test";
  Throttle throttle(concurrency);
  value_type *input_cpu_ptr;
  value_type *output_cpu_ptr;
  checkCudaErrors(cudaMallocHost(
      &input_cpu_ptr, num_tests * mem_pool_->input_size_ * sizeof(value_type)));
  checkCudaErrors(
      cudaMallocHost(&output_cpu_ptr,
                     num_tests * mem_pool_->output_size_ * sizeof(value_type)));
  auto total_start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < num_tests; i++) {
    test_threads_.push_back(std::thread([=, &throttle, mem_pool = mem_pool_] {
      if (i == 100 && bursty == true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        throttle.inc_cur(inc_bursty);
      }
      // if (i == 101 && bursty == true) {
      //     std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      //     throttle.inc_cur(10);
      // }
      // if (i == 102 && bursty == true) {
      //     std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      //     throttle.inc_cur(11);
      // }
      // if (i == 101 && bursty == true) {
      //     std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      //     throttle.dec_cur(inc_bursty);
      // }
      DLOG(INFO) << "waiting to send!!!!!";
      throttle.inc();
      DLOG(INFO) << "begining to send!!!!!";

      send_req(input_cpu_ptr + i * mem_pool->input_size_,
               output_cpu_ptr + i * mem_pool->output_size_, mem_pool);
      throttle.dec();
      DLOG(INFO) << "send ended     !!!!!";
    }));
  }
  for (std::thread &t_j : test_threads_) {
    t_j.join();
  }
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - total_start)
                      .count();
  LOG(INFO) << "total elapsed time: " << duration;
  // std::lock_guard<std::mutex> lock_stop(stop_mutex_);
  // stop_ = true;
}

template <class value_type>
void BatchScheduler<value_type>::start_worker() {
  DLOG(INFO) << "Starting worker!!!";
  for (auto it = worker_pool_->workers_conf_.begin();
       it != worker_pool_->workers_conf_.end(); it++) {
    void *n_key = it->first;
    DLOG(INFO) << "worker: " << n_key;
    Network<value_type> *network = (Network<value_type> *)it->first;
    // network->inference();
    size_t batch_size = it->second->batchsize_;

    // worker_threads_.push_back(
    std::thread([=, worker_pool = worker_pool_, mem_pool = mem_pool_] {
      // DLOG(INFO) << __FILE__ << " " << __LINE__;
      // network->inference();
      checkCudaErrors(cudaStreamWaitEvent(
          network->get_stream(),
          mem_pool->input_event_[it->second->start_index_ + batch_size], 0));
      std::unique_lock<std::mutex> lock_input(it->second->mutex_);
      // DLOG(INFO) << __FILE__ << " " << __LINE__;
      for (;;) {
        // DLOG(INFO) << __FILE__ << " " << __LINE__;
        if (it->second->input_ready_) {
          DLOG(INFO) << "Performing inference !!! Batch size: " << batch_size;
          network->inference(nullptr);

          // DLOG(INFO) << __FILE__ << " " << __LINE__;
          // restore the worker to idle group ans reset it to
          // input not ready
          {
            // TODO update the state in test
            // std::lock_guard<std::mutex>
            // lock_input(*worker_pool->worker_mutex_[n_key]);
            std::lock_guard<std::mutex> lock_comp(mem_pool->comp_mutex_);
            mem_pool->completed_[it->second->start_index_] = true;
          }
          it->second->input_ready_ = false;
          {
            std::lock_guard<std::mutex> lock_idle(worker_pool->idle_mutex_);
            worker_pool->idle_worker_[batch_size].emplace_back(n_key);
            worker_pool->active_requests_ -= batch_size;
          }
          int tmp_start_index = it->second->start_index_;
          DLOG(INFO) << "Start index: " << tmp_start_index;
          for (size_t i = 0; i < batch_size; i++) {
            {
              std::lock_guard<std::mutex> lock_output(
                  *mem_pool->output_mutex_[i + tmp_start_index]);
              mem_pool->output_ready_[i + tmp_start_index] = true;
            }
            mem_pool->output_cond_[i + tmp_start_index]->notify_all();
          }

        } else {
          it->second->cond_.wait(lock_input);
        }
      }
    }).detach();
    //);
  }
}

template <class value_type>
void BatchScheduler<value_type>::schedule(size_t max_requests) {
  worker_pool_->max_requests_ = max_requests;
  // scheduler_thread_ =
  std::thread([&, worker_pool = worker_pool_, mem_pool = mem_pool_] {
    // DLOG(INFO) << __FILE__ << " " << __LINE__;
    for (;;) {
      /* code */

      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      auto waiting = mem_pool->waiting_schedule();
      DLOG(INFO) << __FILE__ << __LINE__ << "scheduling!!!";
      size_t processed = worker_pool->schedule_worker_by_size(
          std::get<0>(waiting), std::get<1>(waiting), std::get<2>(waiting));
      DLOG(INFO) << __FILE__ << __LINE__ << "scheduling!!!";
      mem_pool->update_process(processed);
      DLOG(INFO) << __FILE__ << __LINE__ << "scheduling!!!";
      mem_pool->update_capacity();
      DLOG(INFO) << __FILE__ << __LINE__ << "scheduling!!!";
    }
  }).detach();
  // scheduler_thread_.join();
}

INSTANTIATE_CLASS(BatchScheduler);

}  // namespace ebird
