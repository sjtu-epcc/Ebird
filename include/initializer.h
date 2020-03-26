/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, March 13th 2020, 4:32:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

typedef enum INIT_TYPE {
  _sequential = 0,
  _constant = 1,
  _random = 2,
  _gaussian = 3,
  _xavier = 4,
  _variance = 5
} INIT_TYPE;

namespace ebird {

template <class value_type>
class Initializer {
public:
  virtual void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
                    size_t W) = 0;
  virtual INIT_TYPE get_type() = 0;
};

template <class value_type>
class SequentialInitializer : public Initializer<value_type> {
public:
  SequentialInitializer() {}

  void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
            size_t W) override;

  INIT_TYPE get_type() { return _sequential; }
};

template <class value_type>
class ConstantInitializer : public Initializer<value_type> {
private:
  value_type const_val;

public:
  ConstantInitializer(value_type _const_value) : const_val(_const_value) {}

  void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
            size_t W) override {
    long total = N * C * H * W;
    assert(cpu_ptr != nullptr);
    for (int i = 0; i < total; i++) {
      cpu_ptr[i] = const_val;
      // if(i % 20 == 0) cpu_ptr[i] = i * const_val;
      // else           cpu_ptr[i] = 0;
    }
  }

  INIT_TYPE get_type() { return _constant; }
};

template <class value_type>
class RandomInitializer : public Initializer<value_type> {
public:
  RandomInitializer() {}

  void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
            size_t W) override;

  INIT_TYPE get_type() { return _random; }
};

template <class value_type>
class GaussianInitializer : public Initializer<value_type> {
private:
  value_type mean, std;

public:
  GaussianInitializer(value_type _mean, value_type _std)
      : mean(_mean), std(_std) {}

  void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
            size_t W) override;

  INIT_TYPE get_type() { return _gaussian; }
};

typedef enum fan_type_t {
  FAN_IN = 0,
  FAN_OUT = 1,
  FAN_AVG = 2,
} fan_type_t;

template <class value_type>
class VarianceScalingInitializer : public Initializer<value_type> {
private:
  fan_type_t type;
  value_type factor;
  bool uniform;

public:
  VarianceScalingInitializer(fan_type_t _type, value_type _factor,
                             bool _uniform = false)
      : type(_type), factor(_factor), uniform(_uniform) {}

  void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
            size_t W) override;

  INIT_TYPE get_type() { return _variance; }
};

template <class value_type>
class XavierInitializer : public Initializer<value_type> {
public:
  void call(value_type *cpu_ptr, size_t N, size_t C, size_t H,
            size_t W) override;

  INIT_TYPE get_type() { return _xavier; }
};

}  // namespace ebird
