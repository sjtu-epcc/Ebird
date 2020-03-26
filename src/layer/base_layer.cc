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
#include <layer/base_layer.h>

namespace ebird {

template <typename value_type>
size_t BaseLayer<value_type>::instance_counter_ = 1;

INSTANTIATE_CLASS(BaseLayer);

}  // namespace ebird
