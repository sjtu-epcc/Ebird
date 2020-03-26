/*
 * Created Date: Monday, June 24th 2019, 5:11:50 pm
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

namespace incv4 {

BaseLayer<float>* conv_bn_relu(cudaStream_t& compute_stream,
                               BaseLayer<float>* bottom, size_t outnums,
                               size_t kernel_h, size_t kernel_w, size_t stride,
                               size_t pad_h, size_t pad_w) {
  BaseLayer<float>* conv = (BaseLayer<float>*)new ConvolutionLayer<float>(
      compute_stream, outnums, kernel_h, kernel_w, stride, pad_h, pad_w,
      new XavierInitializer<float>(), false);
  //    BaseLayer<float>* bn     = (BaseLayer<float> *) new
  //    BatchNormalizationLayer<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float>* relu =
      (BaseLayer<float>*)new ActivationLayer<float>(compute_stream);

  bottom->hook(conv);
  conv->hook(relu);
  //    conv->hook(bn);
  //    bn->hook(relu);

  return relu;
}

BaseLayer<float>* inception_a(cudaStream_t& compute_stream,
                              BaseLayer<float>* bottom) {
  BaseLayer<float>* fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  BaseLayer<float>* concat =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  bottom->hook(fork);

  BaseLayer<float>* net;

  // branch 1
  // inception_a1_1x1_2
  net = conv_bn_relu(compute_stream, fork, 96, 1, 1, 1, 0, 0);
  net->hook(concat);

  // branch 2
  // inception_a1_3x3_reduce
  net = conv_bn_relu(compute_stream, fork, 64, 1, 1, 1, 0, 0);
  // inception_a1_3x3
  net = conv_bn_relu(compute_stream, net, 96, 3, 3, 1, 1, 1);
  net->hook(concat);

  // branch 3
  // inception_a1_3x3_2_reduce
  net = conv_bn_relu(compute_stream, fork, 64, 1, 1, 1, 0, 0);
  // inception_a1_3x3_2
  net = conv_bn_relu(compute_stream, net, 96, 3, 3, 1, 1, 1);
  // inception_a1_3x3_3
  net = conv_bn_relu(compute_stream, net, 96, 3, 3, 1, 1, 1);
  net->hook(concat);

  // branch 4
  BaseLayer<float>* pool = (BaseLayer<float>*)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 1, 1, 3, 3, 1, 1);
  fork->hook(pool);
  net = conv_bn_relu(compute_stream, pool, 96, 1, 1, 1, 0, 0);
  net->hook(concat);

  return concat;
}

BaseLayer<float>* inception_b(cudaStream_t& compute_stream,
                              BaseLayer<float>* bottom) {
  BaseLayer<float>* fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  BaseLayer<float>* concat =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  bottom->hook(fork);

  BaseLayer<float>* net;

  // branch 1
  // inception_b1_1x1_2
  net = conv_bn_relu(compute_stream, fork, 384, 1, 1, 1, 0, 0);
  net->hook(concat);

  // branch 2
  // inception_b1_1x7_reduce
  net = conv_bn_relu(compute_stream, fork, 192, 1, 1, 1, 0, 0);
  // inception_b1_1x7
  net = conv_bn_relu(compute_stream, net, 224, 1, 7, 1, 0, 3);
  // inception_b1_7x1
  net = conv_bn_relu(compute_stream, net, 256, 7, 1, 1, 3, 0);
  net->hook(concat);

  // branch 3
  // inception_b1_7x1_2_reduce
  net = conv_bn_relu(compute_stream, fork, 192, 1, 1, 1, 0, 0);
  // inception_b1_7x1_2
  net = conv_bn_relu(compute_stream, net, 192, 7, 1, 1, 3, 0);
  // inception_b1_1x7_2
  net = conv_bn_relu(compute_stream, net, 224, 1, 7, 1, 0, 3);
  // inception_b1_7x1_3
  net = conv_bn_relu(compute_stream, net, 224, 7, 1, 1, 3, 0);
  // inception_b1_1x7_3
  net = conv_bn_relu(compute_stream, net, 256, 1, 7, 1, 0, 3);
  net->hook(concat);

  // branch 4
  // inception_b1_pool_ave
  BaseLayer<float>* pool = (BaseLayer<float>*)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 1, 1, 3, 3, 1, 1);
  // inception_b1_1x1
  fork->hook(pool);
  net = conv_bn_relu(compute_stream, pool, 128, 1, 1, 1, 0, 0);
  net->hook(concat);

  return concat;
}

BaseLayer<float>* inception_c(cudaStream_t& compute_stream,
                              BaseLayer<float>* bottom) {
  BaseLayer<float>* fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  BaseLayer<float>* concat =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  bottom->hook(fork);

  BaseLayer<float>* net;

  // branch 1
  // inception_c1_1x1_2
  net = conv_bn_relu(compute_stream, fork, 256, 1, 1, 1, 0, 0);
  net->hook(concat);

  // branch 3
  // inception_c1_1x1_3
  net = conv_bn_relu(compute_stream, fork, 384, 1, 1, 1, 0, 0);
  BaseLayer<float>* fork_3 =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  net->hook(fork_3);
  // inception_c1_1x3
  net = conv_bn_relu(compute_stream, fork_3, 256, 1, 3, 1, 0, 1);
  net->hook(concat);
  // inception_c1_3x1
  net = conv_bn_relu(compute_stream, fork_3, 256, 3, 1, 1, 1, 0);
  net->hook(concat);

  // branch 4
  // inception_c1_1x1_4
  net = conv_bn_relu(compute_stream, fork, 384, 1, 1, 1, 0, 0);
  // inception_c1_3x1_2
  net = conv_bn_relu(compute_stream, net, 448, 3, 1, 1, 1, 0);
  // inception_c1_1x3_2
  net = conv_bn_relu(compute_stream, net, 512, 1, 3, 1, 0, 1);
  BaseLayer<float>* fork_4 =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  net->hook(fork_4);
  // inception_c1_1x3_3
  net = conv_bn_relu(compute_stream, fork_4, 256, 1, 3, 1, 0, 1);
  net->hook(concat);
  // inception_c1_3x1_3
  net = conv_bn_relu(compute_stream, fork_4, 256, 3, 1, 1, 1, 0);
  net->hook(concat);

  // branch 5
  // inception_c1_pool_ave
  BaseLayer<float>* inception_c1_pool_ave =
      (BaseLayer<float>*)new PoolingLayer<float>(
          compute_stream, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
          CUDNN_NOT_PROPAGATE_NAN, 1, 1, 3, 3, 1, 1);
  fork->hook(inception_c1_pool_ave);
  // inception_c1_1x1
  net = conv_bn_relu(compute_stream, inception_c1_pool_ave, 256, 1, 1, 1, 0, 0);
  net->hook(concat);

  return concat;
}

#define INCEPTION_C 3
#define INCEPTION_H 299
#define INCEPTION_W 299

BaseLayer<float>* inceptionv4(int stream_id, size_t batch_size) {
  cudaStream_t compute_stream = StreamSingleton::get_compute_stream(stream_id);

  size_t C = 3, H = 299, W = 299;

  BaseLayer<float>* input = (BaseLayer<float>*)new DataLayer<float>(
      compute_stream, DATA_INFER, batch_size, C, H, W);

  BaseLayer<float>* conv1_3x3_s2 =
      (BaseLayer<float>*)new ConvolutionLayer<float>(
          compute_stream, 32, 3, 2, 0, 0, new XavierInitializer<float>(),
          false);
  BaseLayer<float>* conv1_3x3_s2_bn =
      (BaseLayer<float>*)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float>* conv1_3x3_s2_relu =
      (BaseLayer<float>*)new ActivationLayer<float>(compute_stream);

  input->hook(conv1_3x3_s2);
  conv1_3x3_s2->hook(conv1_3x3_s2_bn);
  conv1_3x3_s2_bn->hook(conv1_3x3_s2_relu);

  BaseLayer<float>* net;

  // conv2_3x3_s1
  net = conv_bn_relu(compute_stream, conv1_3x3_s2_relu, 32, 3, 3, 1, 0, 0);
  // conv3_3x3_s1
  net = conv_bn_relu(compute_stream, net, 64, 3, 3, 1, 1, 1);
  // stem1
  // fork
  BaseLayer<float>* inception_stem0_fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  // concat
  BaseLayer<float>* inception_stem1 =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  net->hook(inception_stem0_fork);

  // left
  net = conv_bn_relu(compute_stream, inception_stem0_fork, 96, 3, 3, 2, 0, 0);
  net->hook(inception_stem1);

  // right
  BaseLayer<float>* inception_stem1_pool =
      (BaseLayer<float>*)new PoolingLayer<float>(
          compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
          3);

  inception_stem0_fork->hook(inception_stem1_pool);
  inception_stem1_pool->hook(inception_stem1);

  net = inception_stem1;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // stem2

  // fork
  BaseLayer<float>* inception_stem1_fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  // concat
  BaseLayer<float>* inception_stem2 =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  net->hook(inception_stem1_fork);

  // left
  net = conv_bn_relu(compute_stream, inception_stem1_fork, 64, 1, 1, 1, 0, 0);
  net = conv_bn_relu(compute_stream, net, 96, 3, 3, 1, 0, 0);
  net->hook(inception_stem2);

  // right
  net = conv_bn_relu(compute_stream, inception_stem1_fork, 64, 1, 1, 1, 0, 0);
  net = conv_bn_relu(compute_stream, net, 64, 1, 7, 1, 0, 3);
  net = conv_bn_relu(compute_stream, net, 64, 7, 1, 1, 3, 0);
  net = conv_bn_relu(compute_stream, net, 96, 3, 3, 1, 0, 0);
  net->hook(inception_stem2);

  net = inception_stem2;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // stem3
  // fork
  BaseLayer<float>* inception_stem2_fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  // concat
  BaseLayer<float>* inception_stem3 =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  net->hook(inception_stem2_fork);

  // left
  net = conv_bn_relu(compute_stream, inception_stem2_fork, 192, 3, 3, 2, 0, 0);
  net->hook(inception_stem3);

  // right
  BaseLayer<float>* inception_stem3_pool =
      (BaseLayer<float>*)new PoolingLayer<float>(
          compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
          3);
  inception_stem2_fork->hook(inception_stem3_pool);
  inception_stem3_pool->hook(inception_stem3);

  net = inception_stem3;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // inception A

  for (int i = 0; i < 4; ++i) {
    net = inception_a(compute_stream, net);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // reduction_a_concat

  BaseLayer<float>* inception_a4_fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  BaseLayer<float>* reduction_a_concat =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  net->hook(inception_a4_fork);

  // branch 1
  // reduction_a_3x3
  net = conv_bn_relu(compute_stream, inception_a4_fork, 384, 3, 3, 2, 0, 0);
  net->hook(reduction_a_concat);

  // branch 2
  // reduction_a_3x3_2_reduce
  net = conv_bn_relu(compute_stream, inception_a4_fork, 192, 1, 1, 1, 0, 0);
  // reduction_a_3x3_2
  net = conv_bn_relu(compute_stream, net, 224, 3, 3, 1, 1, 1);
  // reduction_a_3x3_3
  net = conv_bn_relu(compute_stream, net, 256, 3, 3, 2, 0, 0);
  net->hook(reduction_a_concat);

  // branch 3
  // reduction_a_pool
  BaseLayer<float>* reduction_a_pool =
      (BaseLayer<float>*)new PoolingLayer<float>(
          compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
          3);
  inception_a4_fork->hook(reduction_a_pool);
  reduction_a_pool->hook(reduction_a_concat);

  net = reduction_a_concat;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // inception B

  for (int i = 0; i < 7; ++i) {
    net = inception_b(compute_stream, net);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // reduction_b_concat
  BaseLayer<float>* inception_b7_fork =
      (BaseLayer<float>*)new ForkLayer<float>(compute_stream);
  BaseLayer<float>* reduction_b_concat =
      (BaseLayer<float>*)new ConcatLayer<float>(compute_stream);

  net->hook(inception_b7_fork);

  // branch 1
  // reduction_b_3x3_reduce
  net = conv_bn_relu(compute_stream, inception_b7_fork, 192, 1, 1, 1, 0, 0);
  // reduction_b_3x3
  net = conv_bn_relu(compute_stream, net, 192, 3, 3, 2, 0, 0);
  net->hook(reduction_b_concat);

  // branch 2
  // reduction_b_1x7_reduce
  net = conv_bn_relu(compute_stream, inception_b7_fork, 256, 1, 1, 1, 0, 0);
  // reduction_b_1x7
  net = conv_bn_relu(compute_stream, net, 256, 1, 7, 1, 0, 3);
  // reduction_b_7x1
  net = conv_bn_relu(compute_stream, net, 320, 7, 1, 1, 3, 0);
  // reduction_b_3x3_2
  net = conv_bn_relu(compute_stream, net, 320, 3, 3, 2, 0, 0);
  net->hook(reduction_b_concat);

  // branch 3
  // reduction_b_pool
  BaseLayer<float>* reduction_b_pool =
      (BaseLayer<float>*)new PoolingLayer<float>(
          compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
          3);
  inception_b7_fork->hook(reduction_b_pool);
  reduction_b_pool->hook(reduction_b_concat);

  net = reduction_b_concat;

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // inception C
  for (int i = 0; i < 3; ++i) {
    net = inception_c(compute_stream, net);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // global pooling
  // TODO : kernel size and stride
  BaseLayer<float>* pool_8x8_s1 = (BaseLayer<float>*)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 1, 1, 8, 8);
  // BaseLayer<float>* drop =
  //     (BaseLayer<float>*)new DropoutLayer<float>(compute_stream,0.2);
  BaseLayer<float>* fc = (BaseLayer<float>*)new FullyConnectedLayer<float>(
      compute_stream, 1000, new XavierInitializer<float>(), true);
  BaseLayer<float>* softmax =
      (BaseLayer<float>*)new SoftmaxLayer<float>(compute_stream);

  net->hook(pool_8x8_s1);
  pool_8x8_s1->hook(fc);
  // drop->hook(fc);
  fc->hook(softmax);

  return input;
}

}  // namespace incv4

}  // namespace ebird