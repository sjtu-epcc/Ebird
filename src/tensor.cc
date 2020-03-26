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
#include <cublas_alias.h>
#include <tensor.h>
#include <util/common.h>
#include <util/mem_util.h>

#include <thread>

namespace ebird {

// PRIVATE METHODS

// PUBLIC METHODS

template <class value_type>
void Tensor<value_type>::GPUtoCPU() {
  /**
   * Sync GPU to CPU
   * state : GPU_FUL
   */
  assert(this->cpu_ptr_ != nullptr);
  assert(this->gpu_ptr_ != nullptr);
  long total = this->N_ * this->C_ * this->H_ * this->W_;
  checkCudaErrors(cudaMemcpy((void *)this->cpu_ptr_, (void *)this->gpu_ptr_,
                             total * sizeof(value_type),
                             cudaMemcpyDeviceToHost));
// #ifndef NDEBUG
//   printf("GPUtoCPU : %p layer %d type %d\n", this, this->get_layer_id(),
//          this->get_type());
// #endif
}

template <class value_type>
void Tensor<value_type>::CPUtoGPU() {
  /**
   * Sync CPU to GPU
   * state : VOID, CPU, GPU_NIL, RECOMPUTE -> GPU_FUL
   */
  long total = this->N_ * this->C_ * this->H_ * this->W_;
  checkCudaErrors(cudaMemcpy((void *)this->gpu_ptr_, (void *)this->cpu_ptr_,
                             total * sizeof(value_type),
                             cudaMemcpyHostToDevice));
}

//------GPU functions-----//
template <class value_type>  // ptr1 = ptr1 + ptr2
void tensor_sum(value_type *ptr1, value_type *ptr2, int size,
                cudaStream_t &cur_stream);

template <class value_type>  // copy ptr1 to ptr2
void tensor_copy(value_type *ptr1, value_type *ptr2, int size,
                 cudaStream_t &cur_stream);

template <class value_type>  // ptr1 = ptr1 * s
void tensor_scale(value_type *ptr1, value_type s, int size,
                  cudaStream_t &cur_stream);
//-----------------------//

template <class value_type>
void Tensor<value_type>::sum(Tensor<value_type> *t, cudaStream_t &cur_stream) {
  size_t len = this->N_ * this->C_ * this ->H_ * this->W_;
  value_type one = 1.0;
  tensor_sum(this->get_gpu_ptr(), t->get_gpu_ptr(), len, cur_stream);
}

template <class value_type>
value_type Tensor<value_type>::squared_sum(cublasHandle_t *handle) {
  size_t len = this->N_ * this->C_ * this->H_ * this->W_;
  value_type squared_sum = 0;
  value_type result = 0;
  CublasDot(handle, this->get_scalar_count(), this->get_gpu_ptr(), 1,
            this->get_gpu_ptr(), 1, &result);
  return result;
}

template <class value_type>
void Tensor<value_type>::copy(Tensor<value_type> *t, cudaStream_t &cur_stream,
                              int src_start_idx, int src_end_idx,
                              int dst_start_idx, int dst_end_idx) {
  size_t len = 0, offset_dst = 0, offset_src = 0;
  if ((src_start_idx == -1) && (src_end_idx == -1) && (dst_start_idx == -1) &&
      (dst_end_idx == -1)) {
    len = this->N_ * this->C_ * this->H_ * this->W_;
  }
  if ((src_start_idx >= 0) && (src_end_idx >= 0)) {
    len = (size_t)(src_end_idx - src_start_idx);
    offset_src = (size_t)src_start_idx;
  }
  if ((dst_start_idx >= 0) && (dst_end_idx >= 0)) {
    if (len != 0) {
      if (len != (size_t)(dst_end_idx - dst_start_idx)) {
        fprintf(stderr,
                "tensor copy size does not match, src len: %zu, dst "
                "len: %d\n",
                len, dst_end_idx - dst_start_idx);
      }
    } else {
      len = (size_t)(dst_end_idx - dst_start_idx);
    }
    offset_dst = (size_t)dst_start_idx;
  }
  // TODO : this memcpy is with error in loss decrease
  //    cudaMemcpy(this->get_gpu_ptr()+offset_dst,
  //    t->get_gpu_ptr()+offset_src, len, cudaMemcpyDeviceToDevice);
  tensor_copy(t->get_gpu_ptr() + offset_src, this->get_gpu_ptr() + offset_dst,
              len, cur_stream);
}

template <class value_type>
void Tensor<value_type>::scale(value_type s, cudaStream_t &cur_stream) {
  size_t len = this->N_ * this->C_ * this->H_ * this->W_;
  tensor_scale(this->get_gpu_ptr(), s, len, cur_stream);
}

template <class value_type>
void Tensor<value_type>::hostRegister() {
  if (this->gpu_ptr_ != nullptr) {
    long total = this->N_ * this->C_ * this->H_ * this->W_;
    checkCudaErrors(cudaHostRegister(this->cpu_ptr_, total * sizeof(value_type),
                                     cudaHostRegisterPortable));
  }
}

#define PRINT_TENSOR
template <class value_type>
void Tensor<value_type>::printTensor(const char *str) {
#ifdef PRINT_TENSOR
  printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
  printf("PRINT OUT TENSOR %p N:%zu C%zu H:%zu W:%zu@:%s\n", this, this->N_,
         this->C_, this->H_, this->W_, str);
  GPUtoCPU();
  for (size_t n = 0; n < this->N_; n++) {
    printf("#################### CPU n:%zu ####################\n", n);
    for (size_t c = 0; c < this->C_; c++) {
      printf("--------c:%zu--------\n", c);
      for (size_t h = 0; h < this->H_; h++) {
        for (size_t w = 0; w < this->W_; w++) {
          // float and double
          printf(" %3.3f, ", this->cpu_ptr_[((n * C_ + c) * H_ + h) * W_ + w]);
        }
        printf("\n");
      }
    }
  }
  printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
#endif
}

template <class value_type>
void Tensor<value_type>::printTensorNoDebug(const char *str) {
  printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
  printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N_, this->C_,
         this->H_, this->W_, str);
  GPUtoCPU();
  for (size_t n = 0; n < this->N_; n++) {
    printf("#################### CPU n:%zu ####################\n", n);
    for (size_t c = 0; c < this->C_; c++) {
      printf("--------c:%zu--------\n", c);
      for (size_t h = 0; h < this->H_; h++) {
        for (size_t w = 0; w < this->W_; w++) {
          // float and double
          printf(" %3.5f ", this->cpu_ptr_[((n * C_ + c) * H_ + h) * W_ + w]);
        }
        printf("\n");
      }
    }
  }
  printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
}

template <class value_type>
void Tensor<value_type>::writeToFile(const char *str) {
  printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
  printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N_, this->C_,
         this->H_, this->W_, str);
  FILE *fp;
  fp = fopen(str, "a");
  GPUtoCPU();
  for (size_t n = 0; n < this->N_; n++) {
    // fprintf(fp, "#################### CPU n:%zu ####################\n",
    // n);
    for (size_t c = 0; c < this->C_; c++) {
      // fprintf(fp, "--------c:%zu--------\n", c);
      for (size_t h = 0; h < this->H_; h++) {
        for (size_t w = 0; w < this->W_; w++) {
          // float and double
          fprintf(fp, "%f ", this->cpu_ptr_[((n * C_ + c) * H_ + h) * W_ + w]);
        }
        // fprintf(fp, "\n");
      }
    }
  }
  fclose(fp);
}

template <class value_type>
void Tensor<value_type>::printTensorFirst(const char *str) {
  printf("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
  printf("PRINT OUT TENSOR N:%zu C:%zu H:%zu W:%zu@:%s\n", this->N_, this->C_,
         this->H_, this->W_, str);
  //        size_t total = this->N*this->C*this->H*this->W;
  GPUtoCPU();
  for (size_t n = 0; n < 1; n++) {
    printf("#################### CPU n:%zu ####################\n", n);
    for (size_t c = 0; c < this->C_; c++) {
      printf("--------c:%zu--------\n", c);
      for (size_t h = 0; h < this->H_; h++) {
        for (size_t w = 0; w < this->W_; w++) {
          // float and double
          printf(" %2.0f ", this->cpu_ptr_[((n * C_ + c) * H_ + h) * W_ + w]);
        }
        printf("\n");
      }
    }
  }
  printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
}

template <class value_type>
void Tensor<value_type>::resizeTensor(size_t n, size_t c, size_t h, size_t w) {
  /**
   * state : not change
   */
  assert(n >= 1);
  assert(c >= 1);
  assert(h >= 1);
  assert(w >= 1);

  //    bool flag = this->gpu_ptr != nullptr;
  freeSpaceGPU();

  //    if (flag) {
  acquireSpaceGPU(n * c * h * w);
  //    }

  freeSpaceCPU();

  // #ifdef TENSOR_REUSE
  //     if (this->data_type_ != CONV_BUFF) {
  //         acquireSpaceCPU(n * c * h * w);
  //     }
  // #else
  acquireSpaceCPU(n * c * h * w);
  // #endif

  this->N_ = n;
  this->C_ = c;
  this->H_ = h;
  this->W_ = w;

  CHECK_GT((int)n, 0);
  CHECK_GT((int)c, 0);
  CHECK_GT((int)h, 0);
  CHECK_GT((int)w, 0);

  checkCUDNN(cudnnDestroyTensorDescriptor(cudnn_tensor_desc_));
  checkCUDNN(cudnnCreateTensorDescriptor(&cudnn_tensor_desc_));
  checkCUDNN(cudnnSetTensor4dDescriptor(this->cudnn_tensor_desc_,
                                        this->cudnn_tensor_format_,
                                        this->cudnn_data_type_, n, c, h, w));
}

template <class value_type>
value_type Tensor<value_type>::get_scalar(const size_t n, const size_t c,
                                          const size_t h, const size_t w) {
  assert(n < N_);
  assert(c < C_);
  assert(h < H_);
  assert(w < W_);
  GPUtoCPU();
  return (this->cpu_ptr_[((n * C_ + c) * H_ + h) * W_ + w]);
}

template <class value_type>
void Tensor<value_type>::set_scalar(const size_t n, const size_t c,
                                    const size_t h, const size_t w,
                                    value_type t) {
  assert(n < N_);
  assert(c < C_);
  assert(h < H_);
  assert(w < W_);
  GPUtoCPU();
  this->cpu_ptr_[((n * C_ + c) * H_ + h) * W_ + w] = t;
  CPUtoGPU();
}

template <class value_type>
void Tensor<value_type>::init(Initializer<value_type> *initializer) {
  initializer->call(this->cpu_ptr_, this->N_, this->C_, this->H_, this->W_);
  CPUtoGPU();

  // NOTE the initializer should be used only once !!!
  //#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
  //    delete initializer;
}

template <class value_type>
void Tensor<value_type>::acquireSpaceGPU(long total) {
  if (gpu_ptr_ != nullptr) {
    return;
  }
  assert(total > 0);

  //    printf("before malloc %zu byte\n", query_free_mem());
  size_t blocks_size = sizeof(value_type) * total;
  // #ifdef TENSOR_REUSE
  //     gpu_ptr = gpu_mem_blocks->getBlockBySize(blocks_size, this,
  //     network_id);
  // #else
  checkCudaErrors(cudaMalloc(&gpu_ptr_, blocks_size));
  // #endif
  //    printf("after malloc %zu byte\n", query_free_mem());
  if (data_type_ != DATA && data_type_ != CONV_BUFF) {
    return;
  }
}

template <class value_type>
void Tensor<value_type>::freeSpaceGPU(MemMode target) {
  if (gpu_ptr_ == nullptr) {
    return;
  }
#ifndef NDEBUG
  printf("free tensor %p layer %d gpu %p target: %d\n", this,
         this->get_layer_id(), gpu_ptr_, target);
#endif
  // FIXME replace following with GpuMemBlocks
  // gfree(gpu_malloc, this->gpu_ptr);

  // gpu_mem_blocks->freeBlockByPtr(gpu_ptr, network_id);
  // NOTE check if there is a need to put gpu_ptr back to nullptr
  // this->gpu_ptr = nullptr;
}

template <class value_type>
void Tensor<value_type>::replace_data(value_type *new_cpu_ptr,
                                      value_type *new_gpu_ptr) {
  if (new_cpu_ptr != nullptr) {
    value_type *old_cpu_ptr = this->cpu_ptr_;
    this->cpu_ptr_ = new_cpu_ptr;
    checkCudaErrors(cudaFreeHost(old_cpu_ptr));

    if (new_gpu_ptr == nullptr) {
      CPUtoGPU();
    }
  }

  if (new_gpu_ptr != nullptr) {
    value_type *old_gpu_ptr = this->gpu_ptr_;
    this->gpu_ptr_ = new_gpu_ptr;

    // remember to free the old ptr
    checkCudaErrors(cudaFree(old_gpu_ptr));
  }
}

template <class value_type>
size_t Tensor<value_type>::tensor_counter_ = 0;

INSTANTIATE_CLASS(Tensor);

}  // namespace ebird
