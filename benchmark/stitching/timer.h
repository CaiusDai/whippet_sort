#ifndef STITCH_TIMER_H_
#define STITCH_TIMER_H_

#pragma once

#include <time.h>

#include <cstdint>
#include <vector>

namespace stitch {

class Timer {
 public:
  // By default use PROCESS_CPU time to avoid frequency change,power scaling and
  // context switch issues.
  Timer(clockid_t clock_type = CLOCK_PROCESS_CPUTIME_ID)
      : clock_type(clock_type) {}
  inline void start() { clock_gettime(clock_type, &start_time); }
  inline void stop() { clock_gettime(clock_type, &end_time); }
  inline double get_elapsed_time_s() {
    return static_cast<double>(end_time.tv_sec - start_time.tv_sec) +
           static_cast<double>(end_time.tv_nsec - start_time.tv_nsec) / 1e9;
  }
  inline double get_elapsed_time_ms() {
    return static_cast<double>(end_time.tv_sec - start_time.tv_sec) * 1e3 +
           static_cast<double>(end_time.tv_nsec - start_time.tv_nsec) / 1e6;
  }

 private:
  timespec start_time;
  timespec end_time;
  clockid_t clock_type;
};  // Class Timer

}  // namespace stitch

#endif  // _STITCH_TIMER_H_