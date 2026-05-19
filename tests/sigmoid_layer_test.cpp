#include "layers/sigmoid_layer.hpp"
#include "layers/layer_factory.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <glog/logging.h>

using namespace mini_inference_engine;

TEST(layer_test, sigmoid_layer) {
    std::shared_ptr<Op> op = std::make_shared<SigmoidOp>();
    std::shared_ptr<Layer> layer = LayerFactory::CreateLayer(op);
    ASSERT_NE(layer, nullptr);

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 2, 2);
    input->Fill(0.f);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    layer->Forward(inputs, outputs);

    ASSERT_EQ(outputs.size(), 1);
    std::shared_ptr<Tensor<float>> output = outputs.at(0);
    ASSERT_NE(output, nullptr);
    
    for (uint32_t i = 0; i < output->Size(); ++i) {
        EXPECT_NEAR(output->Index(i), 0.5f, 1e-5);
    }

    input->At(0, 0, 0) = 1.0f;
    input->At(0, 0, 1) = -1.0f;
    outputs.clear();
    layer->Forward(inputs, outputs);
    output = outputs.at(0);

    EXPECT_NEAR(output->At(0, 0, 0), 1.0f / (1.0f + std::exp(-1)), 1e-5);
    EXPECT_NEAR(output->At(0, 0, 1), 1.0f / (1.0f + std::exp(1)), 1e-5);
}
