/*
 * Created Date: Saturday, June 22nd 2019, 7:25:41 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Monday, March 16th 2020, 1:42:16 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <inceptionv4.h>

using namespace ebird;
using namespace incv4;

DEFINE_int32(threads, 10, "threads use for test");
DEFINE_int32(tests, 1000, "number of tests");
DEFINE_bool(bursty, false, "if provide bursty workload");
DEFINE_int32(range, 0, "range of bursty workload");
DEFINE_int32(throttles, 16, "Throttle of schedule");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int threads = FLAGS_threads;
  int tests = FLAGS_tests;
  bool bursty = FLAGS_bursty;
  int bursty_range = FLAGS_range;
  int throttles = FLAGS_throttles;

  Network<float> *net_1 = new Network<float>();
  Network<float> *net_2 = new Network<float>();
  Network<float> *net_4 = new Network<float>();
  Network<float> *net_8 = new Network<float>();
  Network<float> *net_16 = new Network<float>();
  Network<float> *net_16_1 = new Network<float>();
  // Network<float> *net_32 = new Network<float>();

  BaseLayer<float> *input_1 = inceptionv4(1, 1);
  BaseLayer<float> *input_2 = inceptionv4(2, 2);
  BaseLayer<float> *input_4 = inceptionv4(3, 4);
  BaseLayer<float> *input_8 = inceptionv4(4, 8);
  BaseLayer<float> *input_16 = inceptionv4(5, 16);
  BaseLayer<float> *input_16_1 = inceptionv4(6, 16);
  // BaseLayer<float> *input_32 = inceptionv4(6, 32);
  size_t input_size = INCEPTION_C * INCEPTION_H * INCEPTION_W;
  net_1->fsetup(input_1);
  net_2->fsetup(input_2);
  net_4->fsetup(input_4);
  net_8->fsetup(input_8);
  net_16->fsetup(input_16);
  net_16_1->fsetup(input_16_1);
  // net_32->fsetup(input_32);

  size_t pool_size = 1024;

  size_t output_size = net_1->get_output_size();

  BatchScheduler<float> scheduler(pool_size, input_size, output_size);
  scheduler.add_worker(net_1, 1);
  scheduler.add_worker(net_2, 2);
  scheduler.add_worker(net_4, 4);
  scheduler.add_worker(net_8, 8);
  scheduler.add_worker(net_16, 16);
  scheduler.add_worker(net_16_1, 16);
  // scheduler.add_worker(net_32, 32);
  // DLOG(INFO) << __FILE__ << " " << __LINE__;
  // scheduler.run_net();
  // DLOG(INFO) << __FILE__ << " " << __LINE__;
  // for (auto i = 0; i<100; i++)
  // scheduler.run_net();
  DLOG(INFO) << "-------------------------------";
  scheduler.start_worker();
  DLOG(INFO) << "--------worker started---------";
  DLOG(INFO) << "-------------------------------";
  scheduler.schedule(throttles);
  DLOG(INFO) << "------scheduler started--------";
  scheduler.start_test(tests, threads, bursty, bursty_range);
  return 0;
}