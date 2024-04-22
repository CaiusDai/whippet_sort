#!/bin/bash

root_dir=$(dirname $(realpath $0))
third_party_dir=${root_dir}/third_party
install_dir=${third_party_dir}/install

export CC=clang
export CXX=clang++

PARALLEL=12 # make parallelism
CMAKE_GENERATOR=Ninja
COMMON_CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_C_COMPILER=$CC \
                    -DCMAKE_CXX_COMPILER=$CXX \
                    -DBUILD_SHARED_LIBS=OFF \
                    -DCMAKE_CXX_STANDARD=20 \
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON"

# google-benchmark (use main branch now)
GOOGLE_BENCHMARK_REPO="https://github.com/google/benchmark.git"
GOOGLE_BENCHMARK_SOURCE="google_benchmark"
google_benchmark_install_dir=${install_dir}/google_benchmark

# boost
BOOST_DOWNLOAD="https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz"
BOOST_SOURCE=boost_1_78_0
BOOST_NAME=boost_1_78_0.tar.gz
BOOST_SHA256="94ced8b72956591c4775ae2207a9763d3600b30d9d7446562c552f0a14a63be7"
boost_install_dir=${install_dir}/boost

# double-conversion
DOUBLE_CONVERSION_DOWNLOAD="https://github.com/google/double-conversion/archive/refs/tags/v3.1.6.tar.gz"
DOUBLE_CONVERSION_SOURCE=double-conversion-3.1.6
DOUBLE_CONVERSION_NAME=double_conversion_3.1.6.tar.gz
DOUBLE_CONVERSION_SHA256="8a79e87d02ce1333c9d6c5e47f452596442a343d8c3e9b234e8a62fce1b1d49c"
double_conversion_install_dir=${install_dir}/double_conversion

# libevent
LIBEVENT_DOWNLOAD="https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz"
LIBEVENT_SOURCE="libevent-2.1.12"
LIBEVENT_NAME="libevent-2.1.12.tar.gz"
LIBEVENT_SHA256="92e6de1be9ec176428fd2367677e61ceffc2ee1cb119035037a27d346b0403bb"
libevent_install_dir=${install_dir}/libevent

# fmt
FMTLIB_DOWNLOAD="https://github.com/fmtlib/fmt/archive/refs/tags/8.0.1.tar.gz"
FMTLIB_NAME="fmt-8.0.1.tar.gz"
FMTLIB_SOURCE="fmt-8.0.1"
FMTLIB_SHA256="b06ca3130158c625848f3fb7418f235155a4d389b2abc3a6245fb01cb0eb1e01"
fmt_install_dir=${install_dir}/fmt

# glog
GLOG_DOWNLOAD="https://github.com/google/glog/archive/v0.6.0.tar.gz"
GLOG_NAME="glog-0.6.0.tar.gz"
GLOG_SOURCE="glog-0.6.0"
GLOG_SHA256="8a83bf982f37bb70825df71a9709fa90ea9f4447fb3c099e1d720a439d88bad6"
glog_install_dir=${install_dir}/glog

# folly: deps on boost, double-conversion, libevent, fmt, glog
FOLLY_DOWNLOAD="https://github.com/facebook/folly/archive/refs/tags/v2022.11.14.00.tar.gz"
FOLLY_NAME=folly-v2022.11.14.00.tar.gz
FOLLY_SOURCE=folly-v2022.11.14.00
FOLLY_SHA256="b249436cb61b6dfd5288093565438d8da642b07ae021191a4042b221bc1bdc0e"

# arrow
ARROW_VERSION=release-15.0.0-rc0

check_if_source_exist() {
  if [ -z $1 ]; then
    echo "dir should specified to check if exist." && return 1
  fi
  if [ ! -d $1 ]; then
    echo "$1 does not exist." && return 1
  fi
  return 0
}

download_folly() {
  mkdir -p ${third_party_dir}/tmp_dir
  if wget --no-check-certificate $FOLLY_DOWNLOAD -O ${third_party_dir}/${FOLLY_NAME}; then
    echo "downloaded ${third_party_dir}/${FOLLY_NAME}"
  else
    echo "Failed to downloaded ${third_party_dir}/${FOLLY_NAME}"
    exit 1
  fi
  tar xzf ${third_party_dir}/${FOLLY_NAME} -C ${third_party_dir}/tmp_dir &&
    mv ${third_party_dir}/tmp_dir/* ${third_party_dir}/${FOLLY_SOURCE} || exit
  echo extracted to ${third_party_dir}/${FOLLY_SOURCE}
}

build_folly_deps() {
  pushd ${third_party_dir}/${FOLLY_SOURCE}
  ./build/fbcode_builder/getdeps.py --allow-system-packages build --install-prefix=${install_dir}/folly mvfst
  popd
}

build_folly() {
  check_if_source_exist ${third_party_dir}/${FOLLY_SOURCE} || download_folly
  build_folly_deps
  mkdir -p build
  pushd ${third_party_dir}/${FOLLY_SOURCE}/build
  source ${root_dir}/velox/scripts/setup-helper-functions.sh
  compiler_flags=\"$(get_cxx_flags "unknown")\"
  bash -c "cmake $COMMON_CMAKE_FLAGS \
      -DCMAKE_INSTALL_PREFIX=${install_dir}/folly -DCMAKE_PREFIX_PATH=${install_dir} \
      -DGFLAGS_USE_TARGET_NAMESPACE=TRUE \
      -DBoost_USE_STATIC_RUNTIME=ON \
      -DBOOST_LINK_STATIC=ON \
      -DBUILD_TESTS=OFF \
      -DGFLAGS_NOTHREADS=OFF \
      -DFOLLY_HAVE_INT128_T=ON \
      -DCXX_STD="c++20" \
      -DCMAKE_CXX_FLAGS=${compiler_flags} \
      .."
  make -j ${PARALLEL} install
  popd
}

download_google_benchmark() {
  echo "----- download_google_benchmark -----"
  mkdir -p $third_party_dir
  pushd $third_party_dir
  git clone ${GOOGLE_BENCHMARK_REPO} ${GOOGLE_BENCHMARK_SOURCE}
  popd
}

build_google_benchmark() {
  check_if_source_exist ${third_party_dir}/${GOOGLE_BENCHMARK_SOURCE} || download_google_benchmark
  echo "----- build_google_benchmark -----"
  pushd ${third_party_dir}/${GOOGLE_BENCHMARK_SOURCE}
  cmake -DCMAKE_INSTALL_PREFIX=${google_benchmark_install_dir} \
    -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release \
    -S . -B "__build"
  cmake --build "__build" --config Release --target install
  popd
}

download_arrow() {
  mkdir -p $third_party_dir
  pushd $third_party_dir
  if [ ! -e arrow ]; then git clone https://github.com/apache/arrow.git; fi
  cd arrow
  git checkout ${ARROW_VERSION}
  popd
}

build_arrow() {
  download_arrow
  pushd $third_party_dir/arrow/cpp

  mkdir -p build && cd build

  cmake -G${CMAKE_GENERATOR} ${COMMON_CMAKE_FLAGS} \
    -DCMAKE_INSTALL_PREFIX=${install_dir}/arrow -DCMAKE_PREFIX_PATH=${install_dir} \
    -DBUILD_WARNING_LEVEL=PRODUCTION \
    -DARROW_USE_CCACHE=ON \
    -DARROW_ALTIVEC=OFF \
    -DARROW_DEPENDENCY_USE_SHARED=OFF -DARROW_BOOST_USE_SHARED=OFF -DARROW_BUILD_SHARED=OFF \
    -DARROW_BUILD_STATIC=ON -DARROW_COMPUTE=ON -DARROW_IPC=ON -DARROW_JEMALLOC=OFF \
    -DARROW_SIMD_LEVEL=NONE -DARROW_RUNTIME_SIMD_LEVEL=NONE \
    -DARROW_WITH_BROTLI=OFF \
    -DARROW_WITH_LZ4=ON -Dlz4_SOURCE=BUNDLED -DARROW_WITH_SNAPPY=ON -DSnappy_SOURCE=BUNDLED -DARROW_WITH_ZLIB=ON -DZLIB_SOURCE=BUNDLED \
    -DARROW_WITH_ZSTD=ON -Dzstd_SOURCE=BUNDLED -DThrift_SOURCE=BUNDLED \
    -DARROW_WITH_RE2=OFF \
    -DARROW_WITH_PROTOBUF=OFF -DARROW_WITH_RAPIDJSON=OFF \
    -DARROW_TESTING=ON \
    -DARROW_WITH_UTF8PROC=OFF -DARROW_BUILD_BENCHMARKS=OFF -DARROW_BUILD_EXAMPLES=OFF \
    -DARROW_BUILD_INTEGRATION=OFF \
    -DARROW_CSV=ON -DARROW_JSON=OFF -DARROW_PARQUET=ON \
    -DARROW_FILESYSTEM=ON \
    -DARROW_GCS=OFF -DARROW_S3=OFF -DARROW_HDFS=ON \
    -DARROW_BUILD_UTILITIES=ON -DARROW_BUILD_TESTS=OFF -DARROW_ENABLE_TIMING_TESTS=OFF \
    -DARROW_FUZZING=OFF \
    -DARROW_OPENSSL_USE_SHARED=ON \
    ..
  cmake --build . --config Release -- -j $PARALLEL
  cmake --install .
  popd
}

all() {
  build_google_benchmark
  build_arrow
}

$@
