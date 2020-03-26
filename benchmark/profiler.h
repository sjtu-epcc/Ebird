/*
 * Created Date: Tuesday, March 10th 2020, 11:04:08 am
 * Author: Raphael-Hao
 * -----
 * Last Modified: Saturday, March 14th 2020, 10:04:25 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2020 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once

#include <bits/stdc++.h>
#include <chrono>
#include <gflags/gflags.h>
/** C++ program to Print all
 * combinations of points that
 * can compose a given number
 */
int getCompositions(int sum, std::vector<std::vector<int>> &all_comb) {
  /** array must be static as we want to keep track
  of values stored in arr[] using current calls of
  printCompositions() in function call stack
  */

  int cnt = 0;
  for (int i = 0; i <= 32; i++) {
    if (sum - i * 1 < 0) break;
    for (int j = 0; j <= 16; j++) {
      if (sum - i * 1 - j * 2 < 0) break;
      for (int k = 0; k <= 8; k++) {
        if (sum - i * 1 - j * 2 - k * 4 < 0) break;
        for (int l = 0; l <= 4; l++) {
          if (sum - i * 1 - j * 2 - k * 4 - l * 8 < 0) break;
          for (int m = 0; m <= 2; m++) {
            if (sum - i * 1 - j * 2 - k * 4 - l * 8 - m * 16 < 0) break;
            for (int n = 0; n <= 1; n++) {
              if (sum - i * 1 - j * 2 - k * 4 - l * 8 - m * 16 - n * 32 < 0)
                break;
              if (sum - i * 1 - j * 2 - k * 4 - l * 8 - m * 16 - n * 32 == 0) {
                // std::cout <<
                // std::cout << i << " " << j << " " << k << " "
                //           << l << " " << m << " " << n
                //           << std::endl;
                cnt++;
                all_comb.push_back(std::vector<int>{i, j, k, l, m, n});
              }
            }
          }
        }
      }
    }
  }
  return cnt;
}
/**
 * @brief analyze the data
 *
 * @param time_store
 * @param size
 * @param analysis
 */
void analyzeData(float time_store[], int size, float analysis[], int iters = 1) {
  float time_sum = 0.0f;
  float time_avg = 0.0f;
  float time_max = 0.0f;
  float time_min = time_store[0];
  for (int i = 0; i < size; i++) {
    time_sum = time_sum + time_store[i];
    time_min = time_min < time_store[i] ? time_min : time_store[i];
    time_max = time_max > time_store[i] ? time_max : time_store[i];
  }
  time_avg = time_sum / size;
  analysis[0] = time_min / iters;
  analysis[1] = time_avg / iters;
  analysis[2] = time_max / iters;
}




