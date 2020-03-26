/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:16:11 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <layer/batch_normalization_layer.h>

namespace ebird {

template <class value_type>
void BatchNormalizationLayer<value_type>::forward_setup(
    Registry<value_type> *reg, cudnnHandle_t *cudnn_h) {
#ifndef NDEBUG
  printf("======>setup the forward batch normalization layer:%d\n",
         this->get_id());
#endif
  // hook the output of previous layer
  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *input = reg->get_reg_output(input_l, curt_l);
  assert(input != nullptr);

  Tensor<value_type> *f_out = new Tensor<value_type>(
      input->get_N(), input->get_C(), input->get_H(), input->get_W(),
      reg->get_vector(), DATA, this->get_id());
  // setup the output tensor
  this->set_f_out(f_out, reg);

  // gamma is the weight
  // beta  is the bias
  Tensor<value_type> *gamma = nullptr;
  Tensor<value_type> *beta = nullptr;
  if (this->mode_ == CUDNN_BATCHNORM_SPATIAL) {
    // gamma, beta tensor dims are 1xCx1x1
    //(one value per C-dim normalized over Nx1xHxW subtensors)
    gamma = new Tensor<value_type>(1, input->get_C(), 1, 1, reg->get_vector(),
                                   PARAM, this->get_id());
    beta = new Tensor<value_type>(1, input->get_C(), 1, 1, reg->get_vector(),
                                  PARAM, this->get_id());
    this->resultRunningMean_ =
        new Tensor<value_type>(1, input->get_C(), 1, 1, reg->get_vector(),
                               BN_MEAN_VAR, this->get_id());
    this->resultRunningVariance_ =
        new Tensor<value_type>(1, input->get_C(), 1, 1, reg->get_vector(),
                               BN_MEAN_VAR, this->get_id());
    beta->init(new ConstantInitializer<value_type>(0));
    gamma->init(new ConstantInitializer<value_type>(1));
    //        beta->const_fill(0);
    //        gamma->const_fill(1);

    resultRunningMean_->init(new ConstantInitializer<value_type>(0));
    resultRunningVariance_->init(new ConstantInitializer<value_type>(0));
    //        resultRunningMean->const_fill(0);
    //        resultRunningVariance->const_fill(0);
  } else if (this->mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
    // gamma, beta tensor dims are 1xCxHxWx..
    // (one value per CHW...-slice, normalized over N slice)
    gamma = new Tensor<value_type>(1, input->get_C(), input->get_H(),
                                   input->get_W(), reg->get_vector(), PARAM,
                                   this->get_id());
    beta = new Tensor<value_type>(1, input->get_C(), input->get_H(),
                                  input->get_W(), reg->get_vector(), PARAM,
                                  this->get_id());
    this->resultRunningMean_ = new Tensor<value_type>(
        1, input->get_C(), input->get_H(), input->get_W(), reg->get_vector(),
        BN_MEAN_VAR, this->get_id());
    this->resultRunningVariance_ = new Tensor<value_type>(
        1, input->get_C(), input->get_H(), input->get_W(), reg->get_vector(),
        BN_MEAN_VAR, this->get_id());
    // according to the default value in tensorflow, the beta should
    // initialize as 0
    beta->init(new ConstantInitializer<value_type>(0));
    gamma->init(new ConstantInitializer<value_type>(1));
    //        beta->const_fill(0);
    //        gamma->const_fill(1);

    resultRunningMean_->init(new ConstantInitializer<value_type>(0));
    resultRunningVariance_->init(new ConstantInitializer<value_type>(0));
    //        resultRunningMean->const_fill(0);
    //        resultRunningVariance->const_fill(0);
  }
  assert(gamma != nullptr);
  assert(beta != nullptr);
  // we treat gamma and beta as the layer params
  this->set_bias(beta, reg);
  this->set_weight(gamma, reg);
  // this->set_bias_prev(beta_prev, reg);
  // this->set_weight_prev(gamma_prev, reg);

  // forward hookup check
  assert(this->get_bias() != nullptr);
  assert(this->get_f_out() != nullptr);
  assert(this->get_weight() != nullptr);

  // register the forward dependency
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *t_out = this->get_f_out();
  gamma = this->get_weight();
  beta = this->get_bias();

  reg->register_forward_dependency(this->get_id(), t_in);
  reg->register_forward_dependency(this->get_id(), t_out);
  reg->register_forward_dependency(this->get_id(), gamma);
  reg->register_forward_dependency(this->get_id(), beta);
  reg->register_forward_dependency(this->get_id(), resultRunningMean_);
  reg->register_forward_dependency(this->get_id(), resultRunningVariance_);
}

template <class value_type>
std::vector<value_type> BatchNormalizationLayer<value_type>::forward(
    NetworkStage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h,
    Registry<value_type> *reg) {
  // gamma is the weight
  // beta  is the bias

  int input_l = this->get_input_layer_id();
  int curt_l = this->get_id();
  Tensor<value_type> *t_in = reg->get_reg_output(input_l, curt_l);
  Tensor<value_type> *t_out = this->get_f_out();
  Tensor<value_type> *gamma = this->get_weight();
  Tensor<value_type> *beta = this->get_bias();

  // value_type *running_mean = ;
  // value_type *running_var;

  if (stage == NET_INFER) {
    checkCUDNN(cudnnBatchNormalizationForwardInference(
        *(cudnn_h), this->mode_, &(this->one_), &(this->zero_),
        t_in->get_tensor_desc(), t_in->get_gpu_ptr(), t_out->get_tensor_desc(),
        t_out->get_gpu_ptr(), gamma->get_tensor_desc(), gamma->get_gpu_ptr(),
        beta->get_gpu_ptr(),
        resultRunningMean_
            ->get_gpu_ptr(),  // TO DO, this needs to be serialized as param
                              // but NOT update
        resultRunningVariance_
            ->get_gpu_ptr(),  // TO DO, this needs to be serialized as param
                              // but NOT update
        epsilon_));
  } else {
    NO_TRAIN;
  }
  return std::vector<value_type>();
}

INSTANTIATE_CLASS(BatchNormalizationLayer);

}  // namespace ebird
