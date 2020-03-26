/*
 * Created Date: Saturday, June 22nd 2019, 10:25:25 pm
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
#define ALEXNET_C 3
#define ALEXNET_H 227
#define ALEXNET_W 227
BaseLayer<float> *alexnet(int stream_id, size_t batch_size) {
    // int C = 3;
    // int H = 227, W = 227;
    cudaStream_t compute_stream =
        StreamSingleton::get_compute_stream(stream_id);

    BaseLayer<float> *input = (BaseLayer<float> *)new DataLayer<float>(
        compute_stream, DATA_INFER, batch_size, ALEXNET_C, ALEXNET_H,
        ALEXNET_W);

    BaseLayer<float> *conv_1 =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 1, 11, 4, 0, 0,
            new ConstantInitializer<float>(0.01), true,
            new ConstantInitializer<float>(0.0));

    BaseLayer<float> *relu_1 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    BaseLayer<float> *lrn_1 =
        (BaseLayer<float> *)new LrnLayer<float>(compute_stream);

    BaseLayer<float> *pool_1 =
        (BaseLayer<float> *)new PoolingLayer<float>(
            compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
            3);

    BaseLayer<float> *conv_2 =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 1, 5, 1, 2, 2,
            new ConstantInitializer<float>(0.01), true,
            new ConstantInitializer<float>(0.1));

    BaseLayer<float> *relu_2 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    BaseLayer<float> *lrn_2 =
        (BaseLayer<float> *)new LrnLayer<float>(compute_stream);

    BaseLayer<float> *pool_2 =
        (BaseLayer<float> *)new PoolingLayer<float>(
            compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
            3);

    BaseLayer<float> *conv_3 =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 1, 3, 1, 1, 1,
            new ConstantInitializer<float>(0.01), true,
            new ConstantInitializer<float>(0.0));

    BaseLayer<float> *relu_3 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    BaseLayer<float> *conv_4 =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 1, 3, 1, 1, 1,
            new ConstantInitializer<float>(0.01), true,
            new ConstantInitializer<float>(0.1));
    BaseLayer<float> *relu_4 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);

    BaseLayer<float> *conv_5 =
        (BaseLayer<float> *)new ConvolutionLayer<float>(
            compute_stream, 1, 3, 1, 1, 1,
            new ConstantInitializer<float>(0.01), true,
            new ConstantInitializer<float>(0.1));
    BaseLayer<float> *relu_5 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    BaseLayer<float> *pool_5 =
        (BaseLayer<float> *)new PoolingLayer<float>(
            compute_stream, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3,
            3);

    BaseLayer<float> *full_conn_1 =
        (BaseLayer<float> *)new FullyConnectedLayer<float>(
            compute_stream, 200, new ConstantInitializer<float>(0.005), true,
            new ConstantInitializer<float>(0.1));
    BaseLayer<float> *relu_6 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    BaseLayer<float> *drop_1 =
        (BaseLayer<float> *)new DropoutLayer<float>(compute_stream,
                                                          0.00000001);

    BaseLayer<float> *full_conn_2 =
        (BaseLayer<float> *)new FullyConnectedLayer<float>(
            compute_stream, 200, new ConstantInitializer<float>(0.005), true,
            new ConstantInitializer<float>(0.1));
    BaseLayer<float> *relu_7 = (BaseLayer<float> *)new ActivationLayer<float>(
        compute_stream, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    BaseLayer<float> *drop_2 =
        (BaseLayer<float> *)new DropoutLayer<float>(compute_stream,
                                                          0.00000001);

    BaseLayer<float> *full_conn_3 =
        (BaseLayer<float> *)new FullyConnectedLayer<float>(
            compute_stream, 1000, new ConstantInitializer<float>(0.01), true,
            new ConstantInitializer<float>(0.0));

    BaseLayer<float> *softmax =
        (BaseLayer<float> *)new SoftmaxLayer<float>(
            compute_stream, CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE);

    // setup network
    input->hook(conv_1);
    conv_1->hook(relu_1);
    relu_1->hook(lrn_1);
    lrn_1->hook(pool_1);

    pool_1->hook(conv_2);
    conv_2->hook(relu_2);
    relu_2->hook(lrn_2);
    lrn_2->hook(pool_2);

    pool_2->hook(conv_3);
    conv_3->hook(relu_3);

    relu_3->hook(conv_4);
    conv_4->hook(relu_4);

    relu_4->hook(conv_5);
    conv_5->hook(relu_5);
    relu_5->hook(pool_5);

    pool_5->hook(full_conn_1);
    full_conn_1->hook(relu_6);
    relu_6->hook(full_conn_2);

    // drop_1->hook(full_conn_2);
    full_conn_2->hook(relu_7);
    relu_7->hook(full_conn_3);

    // drop_2->hook(full_conn_3);
    full_conn_3->hook(softmax);

    return input;
}
}  // namespace ebird