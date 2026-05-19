#include "data/tensor.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>

using namespace mini_inference_engine;

TEST(tensor_test, create_1d_tensor) {
    Tensor<float> v1(1, 1, 5);
    Tensor<float> v2(5);
    ASSERT_EQ(v1.Shape().size(), 1);
    ASSERT_EQ(v2.Shape().size(), 1);
}
