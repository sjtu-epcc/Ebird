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
#include <device_atomic_functions.h>
#include <layer/padding_layer.h>
// #include <math_functions.h>
#include <stream_singleton.h>

namespace ebird {

template <class value_type>
__global__ void padding_fkernel(size_t N, size_t C, size_t H, size_t W,
                                size_t pad_C, size_t pad_H, size_t pad_W,
                                const value_type *src, value_type *dst) {
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    size_t CC = C + 2 * pad_C, HH = H + 2 * pad_H, WW = W + 2 * pad_W;

    for (size_t c = 0; c < pad_C; ++c) {
      for (size_t hw = 0; hw < HH * WW; ++hw) {
        dst[(n * CC + c) * HH * WW + hw] = 0.0;
      }
    }

    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < pad_H; ++h) {
        for (size_t w = 0; w < WW; ++w) {
          dst[index(n, c + pad_C, h, w, CC, HH, WW)] = 0.0;
        }
      }

      for (size_t h = 0; h < H; ++h) {
        // pad before w
        for (size_t w = 0; w < pad_W; ++w) {
          dst[index(n, c + pad_C, h + pad_H, w, CC, HH, WW)] = 0;
        }
        // copy it
        for (size_t w = 0; w < W; ++w) {
          dst[index(n, c + pad_C, h + pad_H, w + pad_W, CC, HH, WW)] =
              src[index(n, c, h, w, C, H, W)];
        }
        // pad after
        for (size_t w = 0; w < pad_W; ++w) {
          dst[index(n, c + pad_C, h + pad_H, w + pad_W + W, CC, HH, WW)] = 0;
        }
      }
      // pad after
      for (size_t h = 0; h < pad_H; ++h) {
        for (size_t w = 0; w < WW; ++w) {
          dst[index(n, c + pad_C, h + pad_H + H, w, CC, HH, WW)] = 0.0;
        }
      }
    }

    for (size_t c = 0; c < pad_C; ++c) {
      for (size_t hw = 0; hw < HH * WW; ++hw) {
        dst[(n * CC + c + pad_C + C) * HH * WW + hw] = 0.0;
      }
    }
  }

  __syncthreads();
}

template <class value_type>
void padding_forward(size_t N, size_t C, size_t H, size_t W, size_t pad_C,
                     size_t pad_H, size_t pad_W, const value_type *src,
                     value_type *dst, cudaStream_t &cur_stream) {
  padding_fkernel<value_type><<<(N + 255) / 256, 256, 0, cur_stream>>>(
      N, C, H, W, pad_C, pad_H, pad_W, src, dst);
}

template void padding_forward<float>(size_t, size_t, size_t, size_t, size_t,
                                     size_t, size_t, const float *, float *,
                                     cudaStream_t &cur_stream);

template void padding_forward<double>(size_t, size_t, size_t, size_t, size_t,
                                      size_t, size_t, const double *, double *,
                                      cudaStream_t &cur_stream);

}  // namespace ebird