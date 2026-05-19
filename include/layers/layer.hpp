#ifndef MINI_INFERENCE_ENGINE_LAYERS_LAYER_HPP_
#define MINI_INFERENCE_ENGINE_LAYERS_LAYER_HPP_

#include "data/tensor.hpp"
#include <string>
#include <vector>
#include <memory>

namespace mini_inference_engine {

class Layer {
public:
    explicit Layer(const std::string& layer_name);
    virtual ~Layer() = default;

    virtual void Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs
    );

private:
    std::string layer_name_;
};

}

#endif
