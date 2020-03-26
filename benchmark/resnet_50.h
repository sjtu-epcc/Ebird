/*
 * Created Date: Saturday, June 22nd 2019, 9:28:23 pm
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
  
namespace res50 {

BaseLayer<float> *conv2_x(cudaStream_t &compute_stream,
                          BaseLayer<float> *bottom, bool increase_dim) {
  // fork
  BaseLayer<float> *fork =
      (BaseLayer<float> *)new ForkLayer<float>(compute_stream);
  // join
  BaseLayer<float> *join =
      (BaseLayer<float> *)new JoinLayer<float>(compute_stream);

  bottom->hook(fork);

  // left part
  if (increase_dim) {
    BaseLayer<float> *conv_left =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 256, 1, 1, 0, 0, new XavierInitializer<float>(),
            false);
    BaseLayer<float> *bn_left =
        (BaseLayer<float> *)new BatchNormalizationLayer<float>(
            compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);

    fork->hook(conv_left);
    conv_left->hook(bn_left);
    bn_left->hook(join);
  } else {
    fork->hook(join);
  }

  // right part
  BaseLayer<float> *conv_right1 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 64, 1, 1, 0, 0, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right1 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right1 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right2 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 64, 3, 1, 1, 1, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right2 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right2 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right3 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 256, 1, 1, 0, 0, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right3 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);

  // right part
  fork->hook(conv_right1);
  conv_right1->hook(bn_right1);
  bn_right1->hook(act_right1);
  act_right1->hook(conv_right2);
  conv_right2->hook(bn_right2);
  bn_right2->hook(act_right2);
  act_right2->hook(conv_right3);
  conv_right3->hook(bn_right3);
  bn_right3->hook(join);

  BaseLayer<float> *act = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  join->hook(act);
  return act;
}

BaseLayer<float> *conv3_x(cudaStream_t &compute_stream,
                          BaseLayer<float> *bottom, bool increase_dim) {
  // fork
  BaseLayer<float> *fork =
      (BaseLayer<float> *)new ForkLayer<float>(compute_stream);
  // join
  BaseLayer<float> *join =
      (BaseLayer<float> *)new JoinLayer<float>(compute_stream);

  bottom->hook(fork);

  if (increase_dim) {
    // left part
    BaseLayer<float> *conv_left =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 512, 1, 2, 0, 0, new XavierInitializer<float>(),
            false);
    BaseLayer<float> *bn_left =
        (BaseLayer<float> *)new BatchNormalizationLayer<float>(
            compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
    fork->hook(conv_left);
    conv_left->hook(bn_left);
    bn_left->hook(join);
  } else {
    fork->hook(join);
  }

  // right part

  BaseLayer<float> *conv_right1;

  if (increase_dim) {
    conv_right1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
        compute_stream, 128, 1, 2, 0, 0, new XavierInitializer<float>(), false);
  } else {
    conv_right1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
        compute_stream, 128, 1, 1, 0, 0, new XavierInitializer<float>(), false);
  }

  BaseLayer<float> *bn_right1 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right1 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right2 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 128, 3, 1, 1, 1, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right2 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right2 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right3 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 512, 1, 1, 0, 0, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right3 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);

  // right part
  fork->hook(conv_right1);
  conv_right1->hook(bn_right1);
  bn_right1->hook(act_right1);
  act_right1->hook(conv_right2);
  conv_right2->hook(bn_right2);
  bn_right2->hook(act_right2);
  act_right2->hook(conv_right3);
  conv_right3->hook(bn_right3);
  bn_right3->hook(join);
  BaseLayer<float> *act = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  join->hook(act);
  return act;
}

BaseLayer<float> *conv4_x(cudaStream_t &compute_stream,
                          BaseLayer<float> *bottom, bool increase_dim) {
  // fork
  BaseLayer<float> *fork =
      (BaseLayer<float> *)new ForkLayer<float>(compute_stream);
  // join
  BaseLayer<float> *join =
      (BaseLayer<float> *)new JoinLayer<float>(compute_stream);

  bottom->hook(fork);

  if (increase_dim) {
    // left part
    BaseLayer<float> *conv_left =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 1024, 1, 2, 0, 0, new XavierInitializer<float>(),
            false);
    BaseLayer<float> *bn_left =
        (BaseLayer<float> *)new BatchNormalizationLayer<float>(
            compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
    fork->hook(conv_left);
    conv_left->hook(bn_left);
    bn_left->hook(join);
  } else {
    fork->hook(join);
  }

  // right part

  BaseLayer<float> *conv_right1;

  if (increase_dim) {
    conv_right1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
        compute_stream, 256, 1, 2, 0, 0, new XavierInitializer<float>(), false);
  } else {
    conv_right1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
        compute_stream, 256, 1, 1, 0, 0, new XavierInitializer<float>(), false);
  }

  BaseLayer<float> *bn_right1 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right1 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right2 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 256, 3, 1, 1, 1, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right2 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right2 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right3 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 1024, 1, 1, 0, 0, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right3 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);

  // right part
  fork->hook(conv_right1);
  conv_right1->hook(bn_right1);
  bn_right1->hook(act_right1);
  act_right1->hook(conv_right2);
  conv_right2->hook(bn_right2);
  bn_right2->hook(act_right2);
  act_right2->hook(conv_right3);
  conv_right3->hook(bn_right3);
  bn_right3->hook(join);
  BaseLayer<float> *act = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  join->hook(act);
  return act;
}

BaseLayer<float> *conv5_x(cudaStream_t &compute_stream,
                          BaseLayer<float> *bottom, bool increase_dim) {
  // fork
  BaseLayer<float> *fork =
      (BaseLayer<float> *)new ForkLayer<float>(compute_stream);
  // join
  BaseLayer<float> *join =
      (BaseLayer<float> *)new JoinLayer<float>(compute_stream);

  bottom->hook(fork);

  // left part
  if (increase_dim) {
    BaseLayer<float> *conv_left =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 2048, 1, 2, 0, 0, new XavierInitializer<float>(),
            false);
    BaseLayer<float> *bn_left =
        (BaseLayer<float> *)new BatchNormalizationLayer<float>(
            compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
    fork->hook(conv_left);
    conv_left->hook(bn_left);
    bn_left->hook(join);
  } else {
    fork->hook(join);
  }

  // right part
  BaseLayer<float> *conv_right1;

  if (increase_dim) {
    conv_right1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
        compute_stream, 512, 1, 2, 0, 0, new XavierInitializer<float>(), false);
  } else {
    conv_right1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
        compute_stream, 512, 1, 1, 0, 0, new XavierInitializer<float>(), false);
  }

  BaseLayer<float> *bn_right1 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right1 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right2 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 512, 3, 1, 1, 1, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right2 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_right2 =
      (BaseLayer<float> *)new ActivationLayer<float>(compute_stream);

  BaseLayer<float> *conv_right3 =
      (BaseLayer<float> *)new ConvolutionLayer<float>(
          compute_stream, 2048, 1, 1, 0, 0, new XavierInitializer<float>(),
          false);
  BaseLayer<float> *bn_right3 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);

  // right part
  fork->hook(conv_right1);
  conv_right1->hook(bn_right1);
  bn_right1->hook(act_right1);
  act_right1->hook(conv_right2);
  conv_right2->hook(bn_right2);
  bn_right2->hook(act_right2);
  act_right2->hook(conv_right3);
  conv_right3->hook(bn_right3);
  bn_right3->hook(join);
  BaseLayer<float> *act = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
  join->hook(act);

  return act;
}

#define RESNET_50_C 3
#define RESNET_50_H 227
#define RESNET_50_W 227

BaseLayer<float> *resnet_50(int stream_id, size_t batch_size) {
  cudaStream_t compute_stream = StreamSingleton::get_compute_stream(stream_id);
  size_t C = 3, H = 227, W = 227;

  BaseLayer<float> *input = (BaseLayer<float> *)new DataLayer<float>(
      compute_stream, DATA_INFER, batch_size, C, H, W);

  BaseLayer<float> *conv_1 = (BaseLayer<float> *)new ConvolutionLayer<float>(
      compute_stream, 64, 7, 2, 3, 3, new XavierInitializer<float>(), false);
  BaseLayer<float> *bn_1 =
      (BaseLayer<float> *)new BatchNormalizationLayer<float>(
          compute_stream, CUDNN_BATCHNORM_SPATIAL, 0.001);
  BaseLayer<float> *act_1 = (BaseLayer<float> *)new ActivationLayer<float>(
      compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

  BaseLayer<float> *pool_1 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
  input->hook(conv_1);
  conv_1->hook(bn_1);
  bn_1->hook(act_1);
  act_1->hook(pool_1);

  BaseLayer<float> *net = pool_1;
  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      net = conv2_x(compute_stream, net, true);
    } else {
      net = conv2_x(compute_stream, net, false);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (i == 0) {
      net = conv3_x(compute_stream, net, true);
    } else {
      net = conv3_x(compute_stream, net, false);
    }
  }

  for (int i = 0; i < 6; i++) {
    if (i == 0) {
      net = conv4_x(compute_stream, net, true);
    } else {
      net = conv4_x(compute_stream, net, false);
    }
  }

  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      net = conv5_x(compute_stream, net, true);
    } else {
      net = conv5_x(compute_stream, net, false);
    }
  }

  BaseLayer<float> *pool_2 = (BaseLayer<float> *)new PoolingLayer<float>(
      compute_stream, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
      CUDNN_NOT_PROPAGATE_NAN, 1, 1, 7, 7);

  // 2048 x 1024
  BaseLayer<float> *full_conn_1 =
      (BaseLayer<float> *)new FullyConnectedLayer<float>(
          compute_stream, 1000, new XavierInitializer<float>(), true);
  BaseLayer<float> *softmax = (BaseLayer<float> *)new SoftmaxLayer<float>(
      compute_stream, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);

  net->hook(pool_2);
  pool_2->hook(full_conn_1);
  full_conn_1->hook(softmax);
  return input;
}

}  // namespace res50

}  // namespace ebird