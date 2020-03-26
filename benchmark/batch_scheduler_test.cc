/*
 * Created Date: Wednesday, June 19th 2019, 3:53:45 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:34:15 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <batch_scheduler_test.h>

using namespace ebird;
using namespace incv4;

int main(int argc, char const *argv[]) {
  Network<float> *net_1 = new Network<float>();
  Network<float> *net_2 = new Network<float>();
  Network<float> *net_4 = new Network<float>();
  Network<float> *net_8 = new Network<float>();
  Network<float> *net_16 = new Network<float>();
  Network<float> *net_32 = new Network<float>();
  // BaseLayer<float> *input_1 = alexnet(1, 1);
  // BaseLayer<float> *input_2 = alexnet(2, 2);
  // BaseLayer<float> *input_4 = alexnet(3, 4);
  // BaseLayer<float> *input_8 = alexnet(4, 8);
  // BaseLayer<float> *input_16 = alexnet(5, 16);
  // size_t input_size = ALEXNET_C * ALEXNET_H * ALEXNET_W;

  // BaseLayer<float> *input_1 = resnet_50(1, 1);
  // BaseLayer<float> *input_2 = resnet_50(2, 2);
  // BaseLayer<float> *input_4 = resnet_50(3, 4);
  // BaseLayer<float> *input_8 = resnet_50(4, 8);
  // BaseLayer<float> *input_16 = resnet_50(5, 16);
  // BaseLayer<float> *input_32 = resnet_50(6, 32);
  // size_t input_size = RESNET_50_C * RESNET_50_H * RESNET_50_W;

  BaseLayer<float> *input_1 = inceptionv4(1, 1);
  BaseLayer<float> *input_2 = inceptionv4(2, 2);
  BaseLayer<float> *input_4 = inceptionv4(3, 4);
  BaseLayer<float> *input_8 = inceptionv4(4, 8);
  BaseLayer<float> *input_16 = inceptionv4(5, 16);
  // BaseLayer<float> *input_32 = inceptionv4(6, 32);
  size_t input_size = INCEPTION_C * INCEPTION_H * INCEPTION_W;
  net_1->fsetup(input_1);
  net_2->fsetup(input_2);
  net_4->fsetup(input_4);
  net_8->fsetup(input_8);
  net_16->fsetup(input_16);
  // net_32->fsetup(input_32);

  size_t pool_size = 1024;

  size_t output_size = net_1->get_output_size();

  float *input_gpu_ptr;
  size_t total_input_size = 16 * input_size;
  // checkCudaErrors(cudaMalloc(&input_gpu_ptr, total_input_size *
  // sizeof(float)));

  // net_1->inference(input_gpu_ptr);
  // net_2->inference(input_gpu_ptr);
  // net_4->inference(input_gpu_ptr);
  // net_8->inference(input_gpu_ptr);
  // net_16->inference(input_gpu_ptr);

  // checkCudaErrors(cudaDeviceSynchronize());

  BatchScheduler<float> scheduler(pool_size, input_size, output_size);
  scheduler.add_worker(net_1, 1);
  scheduler.add_worker(net_2, 2);
  scheduler.add_worker(net_4, 4);
  scheduler.add_worker(net_8, 8);
  scheduler.add_worker(net_16, 16);
  // scheduler.add_worker(net_32, 32);
  // DLOG(INFO) << __FILE__ << " " << __LINE__;
  // scheduler.run_net();
  // DLOG(INFO) << __FILE__ << " " << __LINE__;
  // for (auto i = 0; i<100; i++)
  // scheduler.run_net();
  DLOG(INFO) << "-------------------------------";
  scheduler.start_worker();
  DLOG(INFO) << "--------worker started---------";
  DLOG(INFO) << "-------------------------------";
  scheduler.schedule(32);
  DLOG(INFO) << "------scheduler started--------";
  scheduler.start_test(1200, 60);
  return 0;
}
