cmake_minimum_required(VERSION 3.14)
project(mini-inference-engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SP REQUIRED IMPORTED_TARGET sentencepiece)

message(STATUS "Armadillo include: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "SentencePiece include: ${SP_INCLUDE_DIRS}")

add_executable(verify verify.cpp)

target_include_directories(verify PRIVATE 
    ${ARMADILLO_INCLUDE_DIRS}
)

target_compile_definitions(verify PRIVATE 
    GTEST_VERSION="${GTest_VERSION}"
    SP_VERSION="${SP_VERSION}"
)

target_link_libraries(verify PRIVATE 
    ${ARMADILLO_LIBRARIES}
    GTest::gtest_main 
    glog::glog
    PkgConfig::SP
)
