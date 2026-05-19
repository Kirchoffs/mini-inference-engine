#ifndef MINI_INFERENCE_ENGINE_OPS_OP_HPP_
#define MINI_INFERENCE_ENGINE_OPS_OP_HPP_

namespace mini_inference_engine {

enum class OpType {
    kOpUnknown = -1,
    kOpRelu = 0,
    kOpSigmoid  = 1,
    kOpMaxPooling = 2,
};

class Op {
public:
    OpType op_type_ = OpType::kOpUnknown;

    explicit Op(OpType op_type);
    virtual ~Op() = default;
};

}

#endif
