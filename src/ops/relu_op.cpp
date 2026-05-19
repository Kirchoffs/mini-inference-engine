#include "ops/relu_op.hpp"

namespace mini_inference_engine {

ReluOp::ReluOp(float threshold) : Op(OpType::kOpRelu), threshold_(threshold) {
}

void ReluOp::SetThreshold(float threshold) {
    this->threshold_ = threshold;
}

float ReluOp::GetThreshold() const {
    return threshold_;
}

}
