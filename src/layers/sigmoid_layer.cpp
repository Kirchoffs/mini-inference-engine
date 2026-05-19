#include "layers/sigmoid_layer.hpp"
#include "layers/layer_factory.hpp"
#include <cmath>
#include <glog/logging.h>

namespace mini_inference_engine {

SigmoidLayer::SigmoidLayer(const std::shared_ptr<Op>& op) : Layer("Sigmoid") {
    CHECK(op->op_type_ == OpType::kOpSigmoid) << "Operator has a wrong type: " << int(op->op_type_);
    SigmoidOp* sigmoid_op = dynamic_cast<SigmoidOp*>(op.get());

    CHECK(sigmoid_op != nullptr) << "Sigmoid operator is empty";
    this->op_ = std::make_unique<SigmoidOp>();
}

void SigmoidLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs
) {
    CHECK(!inputs.empty());

    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(!inputs.at(i)->Empty());

        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
        std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(*input_data);

        output_data->Data().transform([&](float value) {
            return 1.0f / (1.0f + std::exp(-value));
        });

        outputs.push_back(output_data);
    }
}

std::shared_ptr<Layer> SigmoidLayer::CreateInstance(const std::shared_ptr<Op>& op) {
    std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
    return sigmoid_layer;
}

static LayerRegistrar g_sigmoid_registrar(OpType::kOpSigmoid, SigmoidLayer::CreateInstance);

}
