/*
 * Created Date: Tuesday, March 10th 2020, 10:29:34 am
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:32:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2020 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <inceptionv4.h>
#include <profiler.h>

using namespace ebird;

DEFINE_int32(sum_size, 8, "sum of batch size");

struct calcConvArgs {
  int input_params[4];
  int filter_params[4];
  int conv_params[6];
  int iters;
  float *time_elapsed;
  cudaStream_t stream_id;
  pthread_barrier_t *barrier;
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int sum_size = FLAGS_sum_size;

  std::vector<std::vector<int>> all_comp;
  int comp_cnt = getCompositions(sum_size, all_comp);
  for (auto compositions : all_comp) {
    for (auto composition : compositions) {
    }
  }

  return 0;
}
