#ifndef MINI_INFERENCE_ENGINE_LAYERS_SIGMOID_LAYER_HPP_
#define MINI_INFERENCE_ENGINE_LAYERS_SIGMOID_LAYER_HPP_

#include "layer.hpp"
#include "ops/op.hpp"
#include "ops/sigmoid_op.hpp"
#include <memory>
#include <vector>

namespace mini_inference_engine {

class SigmoidLayer : public Layer {
public:
    explicit SigmoidLayer(const std::shared_ptr<Op>& op);
    ~SigmoidLayer() = default;

    void Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs
    ) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Op>& op);

private:
    std::unique_ptr<SigmoidOp> op_;
};
}

#endif
