#include "layers/relu_layer.hpp"
#include "layers/layer_factory.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>

using namespace mini_inference_engine;

TEST(layer_test, relu_layer) {
    float threshold = 0.4f;
    std::shared_ptr<Op> op = std::make_shared<ReluOp>(threshold);
    std::shared_ptr<Layer> layer = LayerFactory::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 2, 2);
    input->At(0, 0, 0) = 0.2f;
    input->At(0, 0, 1) = 0.6f;
    input->At(0, 1, 0) = 0.4f;
    input->At(0, 1, 1) = -0.2f;

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    layer->Forward(inputs, outputs);

    ASSERT_EQ(outputs.size(), 1);
    std::shared_ptr<Tensor<float>> output = outputs.at(0);
    ASSERT_NE(output, nullptr);
    
    EXPECT_NEAR(output->At(0, 0, 0), 0.f, 1e-5);
    EXPECT_NEAR(output->At(0, 0, 1), 0.6f, 1e-5);
    EXPECT_NEAR(output->At(0, 1, 0), 0.4f, 1e-5);
    EXPECT_NEAR(output->At(0, 1, 1), 0.f, 1e-5);
}
