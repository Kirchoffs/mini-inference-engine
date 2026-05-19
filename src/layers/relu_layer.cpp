#include "layers/relu_layer.hpp"
#include "layers/layer_factory.hpp"
#include "ops/relu_op.hpp"
#include <glog/logging.h>

namespace mini_inference_engine {

ReluLayer::ReluLayer(const std::shared_ptr<Op>& op) : Layer("Relu") {
    CHECK(op->op_type_ == OpType::kOpRelu) << "Operator has a wrong type: " << int(op->op_type_);
    ReluOp* relu_op = dynamic_cast<ReluOp*>(op.get());

    CHECK(relu_op != nullptr) << "Relu operator is empty";
    this->op_ = std::make_unique<ReluOp>(relu_op->GetThreshold());
}

void ReluLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs
) {
    CHECK(!inputs.empty());

    const uint32_t batch_size = inputs.size();
    for (int i = 0; i < batch_size; ++i) {
        CHECK(!inputs.at(i)->Empty());

        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
        std::shared_ptr<Tensor<float>> output_data = input_data->Clone();

        output_data->Data().transform([&](float value) {
            float threshold = op_->GetThreshold();
            if (value >= threshold) {
                return value;
            } else {
                return 0.f;
            }
        });

        outputs.push_back(output_data);
    }
}

std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Op>& op) {
    std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
    return relu_layer;
}

static LayerRegistrar g_relu_registrar(OpType::kOpRelu, ReluLayer::CreateInstance);

}