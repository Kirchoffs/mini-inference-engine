#ifndef MINI_INFERENCE_ENGINE_LAYERS_MAXPOOLING_LAYER_HPP_
#define MINI_INFERENCE_ENGINE_LAYERS_MAXPOOLING_LAYER_HPP_

#include "layer.hpp"
#include "ops/maxpooling_op.hpp"
#include <memory>
#include <vector>

namespace mini_inference_engine {

class MaxPoolingLayer : public Layer {
public:
    explicit MaxPoolingLayer(const std::shared_ptr<Op>& op);
    ~MaxPoolingLayer() = default;

    void Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs
    ) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Op>& op);

private:
    std::unique_ptr<MaxPoolingOp> op_;
};

}

#endif
