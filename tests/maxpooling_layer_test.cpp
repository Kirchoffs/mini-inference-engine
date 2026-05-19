#include "layers/maxpooling_layer.hpp"
#include "layers/layer_factory.hpp"
#include "data/tensor.hpp"
#include <gtest/gtest.h>

using namespace mini_inference_engine;

TEST(layer_test, maxpooling_layer1) {
    auto maxpooling_op = std::make_shared<MaxPoolingOp>(2, 2, 2, 2, 0, 0);
    // MaxPoolingLayer layer(maxpooling_op);
    std::shared_ptr<Layer> layer = LayerFactory::CreateLayer(maxpooling_op);

    auto input = std::make_shared<Tensor<float>>(1, 4, 4);
    input->Fill({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    });

    std::vector<std::shared_ptr<Tensor<float>>> inputs = {input};
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    layer->Forward(inputs, outputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0]->Rows(), 2);
    ASSERT_EQ(outputs[0]->Cols(), 2);

    EXPECT_FLOAT_EQ(outputs[0]->At(0, 0, 0), 6);
    EXPECT_FLOAT_EQ(outputs[0]->At(0, 0, 1), 8);
    EXPECT_FLOAT_EQ(outputs[0]->At(0, 1, 0), 14);
    EXPECT_FLOAT_EQ(outputs[0]->At(0, 1, 1), 16);
}

TEST(layer_test, maxpooling_layer_padding) {
    auto maxpooling_op = std::make_shared<MaxPoolingOp>(3, 3, 2, 2, 1, 1);
    MaxPoolingLayer layer(maxpooling_op);

    auto input = std::make_shared<Tensor<float>>(1, 3, 3);
    input->Fill({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    std::vector<std::shared_ptr<Tensor<float>>> inputs = {input};
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    layer.Forward(inputs, outputs);

    // Input 3x3, Padding 1x1 -> 5x5
    // Pooling 3x3, Stride 2x2
    // Output size: (5 - 3) / 2 + 1 = 2
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0]->Rows(), 2);
    ASSERT_EQ(outputs[0]->Cols(), 2);

    // Padded input:
    // -inf -inf -inf -inf -inf
    // -inf   1    2    3  -inf
    // -inf   4    5    6  -inf
    // -inf   7    8    9  -inf
    // -inf -inf -inf -inf -inf

    // Top-left 3x3 window:
    // -inf -inf -inf
    // -inf   1    2
    // -inf   4    5
    // Max: 5

    // Top-right 3x3 window:
    // -inf -inf -inf
    //  2    3  -inf
    //  5    6  -inf
    // Max: 6

    // Bottom-left 3x3 window:
    // -inf   4    5
    // -inf   7    8
    // -inf -inf -inf
    // Max: 8

    // Bottom-right 3x3 window:
    //  5    6  -inf
    //  8    9  -inf
    // -inf -inf -inf
    // Max: 9

    EXPECT_FLOAT_EQ(outputs[0]->At(0, 0, 0), 5);
    EXPECT_FLOAT_EQ(outputs[0]->At(0, 0, 1), 6);
    EXPECT_FLOAT_EQ(outputs[0]->At(0, 1, 0), 8);
    EXPECT_FLOAT_EQ(outputs[0]->At(0, 1, 1), 9);
}
