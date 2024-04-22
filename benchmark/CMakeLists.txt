# Copyright 2024 Whippet Sort
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_executable(io_encode_benchmark benchmark_util.cc io_encode_benchmark.cc)
target_link_libraries(
  io_encode_benchmark
  GTest::gtest_main
  benchmark::benchmark
  Parquet::parquet_static
  Arrow::arrow_static
  ArrowTesting::arrow_testing_static # for using the random data generator
)
