#!/bin/bash
###
#Created Date: Saturday, March 14th 2020, 4:46:13 pm
#Author: Raphael-Hao
#-----
#Last Modified: Sunday, March 15th 2020, 7:58:40 am
#Modified By: Raphael-Hao
#-----
#Copyright (c) 2020 Happy
#
#Were It to Benefit My Country, I Would Lay Down My Life !
###

all_com=()
profile() {
  sum=$1
  cnt=0
  for ((i = 0; i <= 32; i++)); do
    if [ $((sum - i * 1)) -lt 0 ]; then
      break
    fi
    for ((j = 0; j <= 16; j++)); do
      if [ $((sum - i * 1 - j * 2)) -lt 0 ]; then
        break
      fi
      for ((k = 0; k <= 8; k++)); do
        if [ $((sum - i * 1 - j * 2 - k * 4)) -lt 0 ]; then
          break
        fi
        for ((l = 0; l <= 4; l++)); do
          if [ $((sum - i * 1 - j * 2 - k * 4 - l * 8)) -lt 0 ]; then
            break
          fi
          for ((m = 0; m <= 2; m++)); do
            if [ $((sum - i * 1 - j * 2 - k * 4 - l * 8 - m * 16)) -lt 0 ]; then
              break
            fi
            for ((n = 0; n <= 1; n++)); do
              if [ $((sum - i * 1 - j * 2 - k * 4 - l * 8 - m * 16 - n * 32)) -lt 0 ]; then
                break
              fi
              if [ $((sum - i * 1 - j * 2 - k * 4 - l * 8 - m * 16 - n * 32)) -eq 0 ] && [ "$i" -le 28 ]; then
                # echo "$i $j $k $l $m $n"
                ./build/benchmark/profiler -i "$i" -j "$j" -k "$k" -l "$l" -m "$m" -n "$n" -model "$2" -it "$3" | tail -1
              fi
            done
          done
        done

      done

    done
  done

}
#for ((model = 1; model < 6; model++)); do
#  for bs in {4,8,16,32}; do
#    echo "models: $model; batchsize: $bs"
#    profile "$bs" "$model" 10 >> "$model"_"$bs".csv
#  done
#done
for bs in {4,8,16}; do
  echo "models: 0; batchsize: $bs"
  profile "$bs" 0 10 >> 0_"$bs".csv
done
