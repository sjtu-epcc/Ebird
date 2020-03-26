#pragma once

#include <cmath>

namespace ebird {

template <class value_type>
class MathUtil {
public:
  value_type log_(value_type i) { return std::log(i); }

  value_type abs_(value_type i) { return std::abs(i); }

  value_type max_(value_type a, value_type b) { return std::max(a, b); }
};

}  // namespace ebird
