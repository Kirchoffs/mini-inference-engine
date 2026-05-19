#ifndef MINI_INFERENCE_ENGINE_OPS_RELU_OP_HPP_
#define MINI_INFERENCE_ENGINE_OPS_RELU_OP_HPP_

#include "op.hpp"

namespace mini_inference_engine {

class ReluOp : public Op {
public:
    explicit ReluOp(float thresh);
    ~ReluOp() override = default;

    void SetThreshold(float threshold);
    float GetThreshold() const;

private:
    float threshold_ = 0.f;
};

}

#endif
