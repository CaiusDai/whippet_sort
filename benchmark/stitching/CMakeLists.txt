cmake_minimum_required(VERSION 3.20)
project(Experiemnts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/35d0c365609296fa4730d62057c487e3cfa030ff.zip
)

FetchContent_MakeAvailable(googletest)


add_executable(column_test ${PROJECT_SOURCE_DIR}/column_test.cc)
target_link_libraries(column_test 
    PRIVATE 
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(column_test)

add_executable(time_breakdown ${PROJECT_SOURCE_DIR}/time_breakdown.cc)