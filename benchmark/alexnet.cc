/*
 * Created Date: Saturday, June 22nd 2019, 10:27:21 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 10:44:38 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <inceptionv4.h>
// #include <resnet_50.h>
// #include <resnet_101.h>
// #include <resnet_152.h>
// #include <vgg_16.h>
// #include <vgg_19.h>

#include <chrono>
// #include <ebird.h>
using namespace ebird;

int main(int argc, char const *argv[]) {
  int batch_size = 32;

  size_t input_size = batch_size * INCEPTION_C * INCEPTION_H * INCEPTION_W;
  int stream_id = 1;

  BaseLayer<float> *input_1 = incv4::inceptionv4(stream_id, batch_size);
  BaseLayer<float> *input_2 = incv4::inceptionv4(stream_id + 1, batch_size);

  Network<float> network_1;
  Network<float> network_2;

  network_1.fsetup(input_1);
  network_2.fsetup(input_2);

  float *input_gpu_ptr;
  checkCudaErrors(cudaMalloc(&input_gpu_ptr, input_size * sizeof(float)));
  for (auto i = 0; i < 50; i++) {
    network_1.inference(input_gpu_ptr);
    network_2.inference(input_gpu_ptr);
    checkCudaErrors(
        cudaStreamSynchronize(StreamSingleton::get_compute_stream(stream_id)));
  }
  return 0;
}
