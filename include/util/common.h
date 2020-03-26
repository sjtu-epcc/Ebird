#pragma once

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <glog/logging.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <switch.h>
#include <time.h>

#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "util/log.h"

typedef std::pair<int, int> LayerKey;  // <prev, cur>

typedef enum NetDirect { FORWARD = 0, BACKWARD = 1 } NetDirect;

typedef enum NetworkStage { NET_TRAIN = 0, NET_INFER = 1 } NetworkStage;

typedef enum DataMode { DATA_TRAIN = 0, DATA_INFER = 1 } DataMode;

typedef enum JoinMode { ELEWISE_SUM = 0, ELEWISE_MAX = 1 } JoinMode;

typedef enum StructureType {
  FORK = 0,
  JOIN = 1,
} StructureType;

typedef enum MemMode {
  VOID = 0,
  GPU_NIL = 1,  // gpu with invalid data
  GPU_FUL = 2,  // gpu with valid data
  CPU = 3,
  CPU2GPU = 4,
  GPU2CPU = 5,
} MemMode;

typedef enum LAYER {
  /*---network layers---*/
  CONV = 0,
  POOL = 1,
  ACT = 2,
  BN = 3,
  FC = 4,
  LRN = 5,
  PADDING = 6,
  DATA_L = 7,
  DROPOUT = 8,
  SOFTMAX = 9,
  /*--structure layers--*/
  CONCAT = 10,
  FORK_L = 11,
  JOIN_L = 12
} LAYER;

#define INSTANTIATE_CLASS(classname)   \
  char gInstantiationGuard##classname; \
  template class classname<float>;     \
  template class classname<double>
