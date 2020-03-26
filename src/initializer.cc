/*
 * Created Date: Sunday, May 5th 2019, 1:49:54 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Thursday, March 26th 2020, 11:17:29 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <initializer.h>
#include <math.h>
#include <sys/time.h>
#include <util/common.h>

namespace ebird {

template <class value_type>
void SequentialInitializer<value_type>::call(value_type *cpu_ptr, size_t N,
                                             size_t C, size_t H, size_t W) {
  long total = N * C * H * W;
  assert(cpu_ptr != nullptr);
  for (int i = 0; i < total; i++) {
    cpu_ptr[i] = i;
  }
}

template <class value_type>
void RandomInitializer<value_type>::call(value_type *cpu_ptr, size_t N,
                                         size_t C, size_t H, size_t W) {
  struct timeval time;
  gettimeofday(&time, nullptr);
  srand((unsigned int)((time.tv_sec * 1000) + (time.tv_usec / 1000)));
  long total = N * C * H * W;

  assert(cpu_ptr != nullptr);

  // CPU
  for (int i = 0; i < total; i++) {
    cpu_ptr[i] = ((value_type)rand() / (RAND_MAX)) / 10.0;
  }
}

template <class value_type>
void GaussianInitializer<value_type>::call(value_type *cpu_ptr, size_t N,
                                           size_t C, size_t H, size_t W) {
  size_t total = N * C * H * W;

  unsigned seed =
      (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::normal_distribution<value_type> distribution(this->mean, this->std);
#ifndef NDEBUG
  printf("filling weight in GAUSSIAN[mean:%f, std:%f]\n", mean, std);
#endif
  // CPU
  for (size_t i = 0; i < total; i++) {
    value_type r = distribution(generator);
    assert(!std::isnan(r));
    cpu_ptr[i] = r;
  }
}

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 */
template <class value_type>
void VarianceScalingInitializer<value_type>::call(value_type *cpu_ptr, size_t N,
                                                  size_t C, size_t H,
                                                  size_t W) {
  size_t total = N * C * H * W;
  size_t fan_in = C * H * W;
  size_t fan_out = N * H * W;
  value_type n;
  switch (this->type) {
    case FAN_IN:
      n = fan_in;
      break;
    case FAN_OUT:
      n = fan_out;
      break;
    case FAN_AVG:
      n = value_type(fan_in + fan_out) / value_type(2);
      break;
    default:
      n = value_type(fan_in + fan_out) / value_type(2);
      break;
  }

  if (uniform) {
    // uniform
    value_type scale = sqrt(value_type(3) * factor / n);
    unsigned seed = (unsigned int)std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count();
    std::default_random_engine generator(seed);
    value_type upper = scale;
    value_type lower = -1.0f * scale;
    std::uniform_real_distribution<value_type> distribution(lower, upper);
    #ifndef NDEBUG
    printf(
        "filling weight in variance_scaling_initializer [lower:%f, "
        "upper:%f]\n",
        lower, upper);
#endif
    // CPU
    for (size_t i = 0; i < total; i++) {
      value_type r = distribution(generator);
      assert(!std::isnan(r));
      cpu_ptr[i] = r;
    }
  } else {
    // normal distributed random initialization

    // according to tensorflow.contrib.layers.initializers, the magic number
    // is 1.3
    value_type scale = sqrt(value_type(1.3) * factor / n);
    (new GaussianInitializer<value_type>(0.0, scale))
        ->call(cpu_ptr, N, C, H, W);
  }
}

template <class value_type>
void XavierInitializer<value_type>::call(value_type *cpu_ptr, size_t N,
                                         size_t C, size_t H, size_t W) {
  (new VarianceScalingInitializer<value_type>(fan_type_t::FAN_AVG, 1.0, true))
      ->call(cpu_ptr, N, C, H, W);
}

INSTANTIATE_CLASS(SequentialInitializer);

INSTANTIATE_CLASS(RandomInitializer);

INSTANTIATE_CLASS(GaussianInitializer);

INSTANTIATE_CLASS(VarianceScalingInitializer);

INSTANTIATE_CLASS(XavierInitializer);

}  // namespace ebird
