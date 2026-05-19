#ifndef MINI_INFERENCE_ENGINE_LAYERS_RELU_LAYER_HPP_
#define MINI_INFERENCE_ENGINE_LAYERS_RELU_LAYER_HPP_

#include "layer.hpp"
#include "ops/relu_op.hpp"
#include <memory>
#include <vector>

namespace mini_inference_engine {

class ReluLayer : public Layer {
public:
    explicit ReluLayer(const std::shared_ptr<Op>& op);
    ~ReluLayer() = default;

    void Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs
    ) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Op>& op);

private:
    std::unique_ptr<ReluOp> op_;
};

}

#endif
