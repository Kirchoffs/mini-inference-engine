#ifndef MINI_INFERENCE_ENGINE_OPS_MAXPOOLING_OP_HPP_
#define MINI_INFERENCE_ENGINE_OPS_MAXPOOLING_OP_HPP_

#include "op.hpp"
#include <cstdint>

namespace mini_inference_engine {

class MaxPoolingOp : public Op {
public:
    explicit MaxPoolingOp(
        uint32_t pooling_h, uint32_t pooling_w,
        uint32_t stride_h, uint32_t stride_w,
        uint32_t padding_h, uint32_t padding_w
    );

    void set_pooling_h(uint32_t pooling_height);
    void set_pooling_w(uint32_t pooling_width);

    void set_stride_h(uint32_t stride_height);
    void set_stride_w(uint32_t stride_width);

    void set_padding_h(uint32_t padding_height);
    void set_padding_w(uint32_t padding_width);

    uint32_t padding_height() const;
    uint32_t padding_width() const;

    uint32_t stride_height() const;
    uint32_t stride_width() const;

    uint32_t pooling_height() const;
    uint32_t pooling_width() const;

private:
    uint32_t pooling_h_;
    uint32_t pooling_w_;
    uint32_t stride_h_;
    uint32_t stride_w_;
    uint32_t padding_h_;
    uint32_t padding_w_;
};

}

#endif
