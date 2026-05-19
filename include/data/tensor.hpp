#ifndef MINI_INFERENCE_ENGINE_DATA_TENSOR_HPP_
#define MINI_INFERENCE_ENGINE_DATA_TENSOR_HPP_

#include <memory>
#include <vector>
#include <armadillo>

namespace mini_inference_engine {

template<typename T>
class Tensor {
};

template<>
class Tensor<uint8_t> {
};

template<>
class Tensor<float> {
public:
    explicit Tensor() = default;
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    explicit Tensor(uint32_t size);
    explicit Tensor(uint32_t rows, uint32_t cols);
    Tensor(const Tensor& tensor);
    Tensor<float>& operator=(const Tensor& tensor);

    uint32_t Rows() const;
    uint32_t Cols() const;
    uint32_t Channels() const;
    uint32_t Size() const;
    std::vector<uint32_t> Shape() const;
    bool Empty() const;

    std::shared_ptr<Tensor<float>> Clone();

    arma::fcube& Data();
    const arma::fcube& Data() const;

    std::vector<float> Values(bool row_major = true) const;
    
    arma::fmat& At(uint32_t channel);
    const arma::fmat& At(uint32_t channel) const;
    
    float& Index(uint32_t offset);
    float Index(uint32_t offset) const;

    float& At(uint32_t channel, uint32_t row, uint32_t col);
    float At(uint32_t channel, uint32_t row, uint32_t col) const;

    void SetData(const arma::fcube& data);
    void Fill(float value);
    void Fill(const std::vector<float>& values, bool row_major = true);
    void Ones();
    void Rand();

    void Padding(const std::vector<uint32_t>& pads, float padding_value);
    void Flatten(bool row_major = false);

    void Show();

private:
    std::vector<uint32_t> shape_;
    arma::fcube data_;
};

}

#endif