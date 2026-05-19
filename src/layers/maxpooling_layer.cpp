#include "layers/maxpooling_layer.hpp"
#include "layers/layer_factory.hpp"
#include <glog/logging.h>

namespace mini_inference_engine {

MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Op>& op) : Layer("MaxPooling") {
    CHECK(op->op_type_ == OpType::kOpMaxPooling) << "Operator has a wrong type: " << int(op->op_type_);
    MaxPoolingOp* maxpooling_op = dynamic_cast<MaxPoolingOp*>(op.get());

    CHECK(maxpooling_op != nullptr) << "MaxPooling operator is empty";
    this->op_ = std::make_unique<MaxPoolingOp>(
        maxpooling_op->pooling_height(), maxpooling_op->pooling_width(),
        maxpooling_op->stride_height(), maxpooling_op->stride_width(),
        maxpooling_op->padding_height(), maxpooling_op->padding_width()
    );
}

void MaxPoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs
) {
    CHECK(!inputs.empty());
    const uint32_t batch_size = inputs.size();

    const uint32_t pooling_h = op_->pooling_height();
    const uint32_t pooling_w = op_->pooling_width();
    const uint32_t stride_h = op_->stride_height();
    const uint32_t stride_w = op_->stride_width();
    const uint32_t padding_h = op_->padding_height();
    const uint32_t padding_w = op_->padding_width();

    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
        CHECK(!input_data->Empty());

        std::shared_ptr<Tensor<float>> input_data_padded = input_data;
        if (padding_h > 0 || padding_w > 0) {
            input_data_padded = input_data->Clone();
            input_data_padded->Padding({padding_h, padding_h, padding_w, padding_w}, std::numeric_limits<float>::lowest());
        }

        const uint32_t input_h = input_data_padded->Rows();
        const uint32_t input_w = input_data_padded->Cols();
        const uint32_t input_c = input_data_padded->Channels();

        const uint32_t output_h = uint32_t(std::floor((input_h - pooling_h) / stride_h + 1));
        const uint32_t output_w = uint32_t(std::floor((input_w - pooling_w) / stride_w + 1));

        std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(input_c, output_h, output_w);

        for (uint32_t c = 0; c < input_c; ++c) {
            const arma::fmat& input_channel = input_data_padded->At(c);
            arma::fmat& output_channel = output_data->At(c);
            for (uint32_t h = 0; h < output_h; ++h) {
                for (uint32_t w = 0; w < output_w; ++w) {
                    float max_value = std::numeric_limits<float>::lowest();
                    for (uint32_t ph = 0; ph < pooling_h; ++ph) {
                        for (uint32_t pw = 0; pw < pooling_w; ++pw) {
                            float value = input_channel.at(h * stride_h + ph, w * stride_w + pw);
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                    output_channel.at(h, w) = max_value;
                }
            }
        }

        outputs.push_back(output_data);
    }
}

std::shared_ptr<Layer> MaxPoolingLayer::CreateInstance(const std::shared_ptr<Op>& op) {
    std::shared_ptr<Layer> maxpooling_layer = std::make_shared<MaxPoolingLayer>(op);
    return maxpooling_layer;
}

static LayerRegistrar g_maxpooling_registrar(OpType::kOpMaxPooling, MaxPoolingLayer::CreateInstance);

}
