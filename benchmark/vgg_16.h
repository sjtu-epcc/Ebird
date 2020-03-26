/*
 * Created Date: Wednesday, June 26th 2019, 2:21:29 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 10:44:38 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once
#include <ebird.h>
namespace ebird {

namespace vgg16 {

#define VGG_16_C 3
#define VGG_16_H 277
#define VGG_16_W 277

BaseLayer<float> *vgg_16(int stream_id, size_t batch_size) {
  cudaStream_t compute_stream = StreamSingleton::get_compute_stream(stream_id);
  BaseLayer<float> *input = (BaseLayer<float> *)new DataLayer<float>(
      compute_stream, DATA_INFER, batch_size, VGG_16_C, VGG_16_H, VGG_16_W);

  // if the dims of H,W after conv is not reduced, pad with half the filter
  // sizes (round down). 3/2 = 1.5 = 1;
  BaseLayer<float> *conv1_1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 64, 3, 1, 1, 1, new GaussianInitializer<float>(0, 0.01),
      true);
  BaseLayer<float> *act1_1 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv1_2 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 64, 3, 1, 1, 1, new GaussianInitializer<float>(0, 0.01),
      true);
  BaseLayer<float> *act1_2 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *pool_1 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);

  BaseLayer<float> *conv2_1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 128, 3, 1, 1, 1, new GaussianInitializer<float>(0, 0.01),
      true);
  BaseLayer<float> *act2_1 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv2_2 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 128, 3, 1, 1, 1, new GaussianInitializer<float>(0, 0.01),
      true);
  BaseLayer<float> *act2_2 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *pool_2 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);

  BaseLayer<float> *conv3_1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 256, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act3_1 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv3_2 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 256, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act3_2 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv3_3 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 256, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act3_3 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *pool_3 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);

  BaseLayer<float> *conv4_1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act4_1 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv4_2 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act4_2 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv4_3 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act4_3 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

  BaseLayer<float> *pool_4 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);

  BaseLayer<float> *conv5_1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act5_1 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv5_2 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act5_2 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *conv5_3 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(), true);
  BaseLayer<float> *act5_3 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  BaseLayer<float> *pool_5 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);

  BaseLayer<float> *full_conn_1 =
      (BaseLayer<float> *)new FullyConnectedLayer<float>(
          compute_stream, 4096, new GaussianInitializer<float>(0, 0.01), true);
  BaseLayer<float> *relu6 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

  BaseLayer<float> *full_conn_2 =
      (BaseLayer<float> *)new FullyConnectedLayer<float>(
          compute_stream, 4096, new GaussianInitializer<float>(0, 0.01), true);
  BaseLayer<float> *relu7 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

  BaseLayer<float> *full_conn_3 =
      (BaseLayer<float> *)new FullyConnectedLayer<float>(
          compute_stream, 1000, new GaussianInitializer<float>(0, 0.01), true);
  BaseLayer<float> *softmax = (BaseLayer<float> *)new SoftmaxLayer<float>(
      compute_stream, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

  // setup network
  input->hook(conv1_1);

  conv1_1->hook(act1_1);
  act1_1->hook(conv1_2);
  conv1_2->hook(act1_2);
  act1_2->hook(pool_1);
  pool_1->hook(conv2_1);

  conv2_1->hook(act2_1);
  act2_1->hook(conv2_2);
  conv2_2->hook(act2_2);
  act2_2->hook(pool_2);
  pool_2->hook(conv3_1);

  conv3_1->hook(act3_1);
  act3_1->hook(conv3_2);
  conv3_2->hook(act3_2);
  act3_2->hook(conv3_3);
  conv3_3->hook(act3_3);
  act3_3->hook(pool_3);
  pool_3->hook(conv4_1);

  conv4_1->hook(act4_1);
  act4_1->hook(conv4_2);
  conv4_2->hook(act4_2);
  act4_2->hook(conv4_3);
  conv4_3->hook(act4_3);
  act4_3->hook(pool_4);
  pool_4->hook(conv5_1);

  conv5_1->hook(act5_1);
  act5_1->hook(conv5_2);
  conv5_2->hook(act5_2);
  act5_2->hook(conv5_3);
  conv5_3->hook(act5_3);
  act5_3->hook(pool_5);
  pool_5->hook(full_conn_1);

  full_conn_1->hook(relu6);
  relu6->hook(full_conn_2);
  full_conn_2->hook(relu7);
  relu7->hook(full_conn_3);

  full_conn_3->hook(softmax);
  return input;
}

}  // namespace vgg16

}  // namespace ebird