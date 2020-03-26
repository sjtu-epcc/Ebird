/*
 * Created Date: Wednesday, March 11th 2020, 10:39:14 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Saturday, March 14th 2020, 10:04:39 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2020 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <inceptionv4.h>
#include <profiler.h>
#include <resnet_101.h>
#include <resnet_152.h>
#include <resnet_50.h>
#include <vgg_16.h>
#include <vgg_19.h>
#include <thread>

DEFINE_int32(i, 1, "num of batch size 1");
DEFINE_int32(j, 1, "num of batch size 2");
DEFINE_int32(k, 1, "num of batch size 4");
DEFINE_int32(l, 1, "num of batch size 8");
DEFINE_int32(m, 1, "num of batch size 16");
DEFINE_int32(n, 1, "num of batch size 32");
DEFINE_int32(model, 0,
             "model id, 0: inceptionv4; 1: resnet 50; 2: resnet 101; 3: resnet "
             "152; 4: vgg 16; 5: vgg19");
DEFINE_int32(it, 10, "test iterations");

using namespace ebird;

void profInstance(int comp[], int model, int iters) {
  int batch_sizes[6] = {1, 2, 4, 8, 16, 32};
  pthread_barrier_t barrier;
  int streams = 0;
  for (size_t i = 0; i < 6; i++) {
    streams += comp[i];
  }
  pthread_barrier_init(&barrier, nullptr, streams);
  std::vector<std::thread> threads;
  float time_store[streams];
  float results[3];
  int stream_cnt = 0;
  for (size_t i = 0; i < 6; i++) {
    for (int j = 0; j < comp[i]; j++) {
      int stream_id = stream_cnt;
      cudaStream_t prof_stream = StreamSingleton::get_compute_stream(stream_id);
      BaseLayer<float>* input_layer;
      Network<float>* network = new Network<float>();
      switch (FLAGS_model) {
        case 0:
          input_layer = incv4::inceptionv4(stream_cnt, batch_sizes[i]);
          break;
        case 1:
          input_layer = res50::resnet_50(stream_cnt, batch_sizes[i]);
          break;
        case 2:
          input_layer = res101::resnet_101(stream_cnt, batch_sizes[i]);
          break;
        case 3:
          input_layer = res152::resnet_152(stream_cnt, batch_sizes[i]);
          break;
        case 4:
          input_layer = vgg16::vgg_16(stream_cnt, batch_sizes[i]);
          break;
        case 5:
          input_layer = vgg19::vgg_19(stream_cnt, batch_sizes[i]);
          break;
        default:
          std::cout << " input layer not initiate" << std::endl;
          std::abort();
          break;
      }
      network->fsetup(input_layer);
      std::cout<< "setup ended: "<< stream_cnt << std::endl;
      threads.emplace_back([&, network, stream_id, prof_stream] {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        checkCudaErrors(cudaEventCreate(&start_event));
        checkCudaErrors(cudaEventCreate(&end_event));
        pthread_barrier_wait(&barrier);
        checkCudaErrors(cudaEventRecord(start_event, prof_stream));

        for (auto iter = 0; iter < iters; iter++) {
          network->inference();
        }
        checkCudaErrors(cudaEventRecord(end_event, prof_stream));
        checkCudaErrors(cudaEventSynchronize(end_event));

        DLOG(INFO) << "net profile completed:" << stream_cnt;
        checkCudaErrors(cudaEventElapsedTime(&time_store[stream_id],
                                             start_event, end_event));
        std::cout << "stream id: " << stream_id
                  << "elapased time: " << time_store[stream_id] << std::endl;
      });
      stream_cnt++;
      // threads.emplace_back(std::thread([&] { network->inference(); }));
    }
  }
  for (auto& th : threads) {
    th.join();
  }
  for (auto i = 0; i < streams; i++) {
    std::cout << time_store[i] << ",";
  }
  std::cout << std::endl;

  analyzeData(time_store, streams, results, FLAGS_it);
  for (auto i = 0; i < 6; i++) {
    std::cout << comp[i] << ",";
  }
  std::cout << results[0] << "," << results[1] << "," << results[2]
            << std::endl;
  // cudaEvent_t start_event;
  // cudaEvent_t end_event;
  // checkCudaErrors(cudaEventCreate(&start_event));
  // checkCudaErrors(cudaEventCreate(&end_event));

  // checkCudaErrors(cudaDeviceSynchronize());
  // pthread_barrier_wait(barrier);
  // for (int i = 0; i < iters; i++) {
  //   prof_conf.network->inference();
  // }
  // checkCudaErrors(cudaEventRecord(end_event, stream_id));
  // checkCudaErrors(cudaEventSynchronize(end_event));
  // checkCudaErrors(
  //     cudaEventElapsedTime(prof_conf.time_elapsed, start_event,
  //     end_event));
}

int main(int argc, char const* argv[]) {
  char** argv_ = const_cast<char**>(argv);
  ::google::ParseCommandLineFlags(&argc, &argv_, true);
  int comp[6] = {FLAGS_i, FLAGS_j, FLAGS_k, FLAGS_l, FLAGS_m, FLAGS_n};
  profInstance(comp, FLAGS_model, FLAGS_it);
  // int comp_cnt = getCompositions(FLAGS_bs, all_comp);

  // for (auto prof_conf : prof_confs) {
  //   delete prof_conf.network;
  // }

  return 0;
}
