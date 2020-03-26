#pragma once
#include <assert.h>
#include <layer/base_structure_layer.h>
#include <switch.h>
#include <tensor.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
class JoinLayer : BaseStructureLayer<value_type> {
private:
  // cudnn setup
  const value_type zero_;
  const value_type one_;

public:
  JoinLayer(cudaStream_t layer_stream)
      : one_(1),
        zero_(0),
        BaseStructureLayer<value_type>(JOIN_L, layer_stream) {
    this->set_structure_type(JOIN);
  }

  ~JoinLayer() {}

  void forward_setup(Registry<value_type> *reg, cudnnHandle_t *cudnn_h);

  std::vector<value_type> forward(NetworkStage stage, cublasHandle_t *cublas_h,
                                  cudnnHandle_t *cudnn_h,
                                  Registry<value_type> *reg);

  void gen_description(char *buff, size_t *len_in_byte) {
    this->gen_meta_description(buff, len_in_byte);
  }

  inline cudaStream_t &get_stream() { return this->_get_stream(); }
};

}  // namespace ebird
