#include "data/tensor.hpp"
#include <glog/logging.h>

namespace mini_inference_engine {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        shape_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        shape_ = std::vector<uint32_t>{rows, cols};
    } else {
        shape_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

Tensor<float>::Tensor(uint32_t size) {
    data_ = arma::fcube(1, size, 1);
    shape_ = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
    data_ = arma::fcube(rows, cols, 1);
    shape_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(const Tensor& tensor)
    : data_(tensor.data_), shape_(tensor.shape_) {
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
    if (this != &tensor) {
        this->data_ = tensor.data_;
        this->shape_ = tensor.shape_;
    }
    
    return *this;
}

uint32_t Tensor<float>::Rows() const {
    CHECK(!this->data_.empty());
    return this->data_.n_rows;
}

uint32_t Tensor<float>::Cols() const {
    CHECK(!this->data_.empty());
    return this->data_.n_cols;
}

uint32_t Tensor<float>::Channels() const {
    CHECK(!this->data_.empty());
    return this->data_.n_slices;
}

uint32_t Tensor<float>::Size() const {
    CHECK(!this->data_.empty());
    return this->data_.size();
}

std::vector<uint32_t> Tensor<float>::Shape() const {
    CHECK(!this->data_.empty());
    return this->shape_;
}

bool Tensor<float>::Empty() const {
    return this->data_.empty();
}

std::shared_ptr<Tensor<float>> Tensor<float>::Clone() {
    return std::make_shared<Tensor<float>>(*this);
}

arma::fcube& Tensor<float>::Data() {
    return this->data_;
}

const arma::fcube& Tensor<float>::Data() const {
    return this->data_;
}

std::vector<float> Tensor<float>::Values(bool row_major) const {
    CHECK_EQ(this->data_.empty(), false);

    std::vector<float> values(this->data_.size());
    if (row_major) {
        uint32_t index = 0;
        for (uint32_t i = 0; i < this->data_.n_slices; ++i) {
            arma::fmat channel = this->data_.slice(i).t();
            std::copy(channel.begin(), channel.end(), values.begin() + index);
            index += channel.size();
        }
        CHECK_EQ(index, values.size());
    } else {
        std::copy(
            this->data_.memptr(), 
            this->data_.memptr() + this->data_.size(),
            values.begin()
        );
    }

    return values;
}

arma::fmat& Tensor<float>::At(uint32_t channel) {
    CHECK_LT(channel, this->Channels());
    return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::At(uint32_t channel) const {
    CHECK_LT(channel, this->Channels());
    return this->data_.slice(channel);
}

float& Tensor<float>::Index(uint32_t offset) {
    CHECK_LT(offset, this->Size());
    return this->data_.at(offset);
}

float Tensor<float>::Index(uint32_t offset) const {
    CHECK_LT(offset, this->Size());
    return this->data_.at(offset);
}

float& Tensor<float>::At(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK_LT(row, this->Rows());
    CHECK_LT(col, this->Cols());
    CHECK_LT(channel, this->Channels());
    return this->data_.at(row, col, channel);
}

float Tensor<float>::At(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK_LT(row, this->Rows());
    CHECK_LT(col, this->Cols());
    CHECK_LT(channel, this->Channels());
    return this->data_.at(row, col, channel);
}

void Tensor<float>::SetData(const arma::fcube& data) {
    CHECK(data.n_rows == this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
    CHECK(data.n_cols == this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
    CHECK(data.n_slices == this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
    this->data_ = data;
}

void Tensor<float>::Fill(float value) {
    CHECK(!this->data_.empty());
    this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
    CHECK(!this->data_.empty());
    CHECK_EQ(values.size(), this->data_.size());

    if (row_major) {
        const uint32_t rows = this->Rows();
        const uint32_t cols = this->Cols();
        const uint32_t planes = rows * cols;
        const uint32_t channels = this->data_.n_slices;

        for (uint32_t i = 0; i < channels; ++i) {
            auto& channel_data = this->data_.slice(i);
            arma::fmat channel_data_t = arma::fmat(const_cast<float*>(values.data()) + i * planes, cols, rows);
            channel_data = channel_data_t.t();
        }
    } else {
        std::copy(values.begin(), values.end(), this->data_.memptr());
    }
}

void Tensor<float>::Ones() {
    CHECK(!this->data_.empty());
    this->Fill(1.f);
}

void Tensor<float>::Rand() {
    CHECK(!this->data_.empty());
    this->data_.randn();
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads, float padding_value) {
    CHECK(!this->data_.empty());
    CHECK_EQ(pads.size(), 4);

    uint32_t padding_rows_up = pads.at(0);
    uint32_t padding_rows_bottom = pads.at(1);
    uint32_t padding_cols_left = pads.at(2);
    uint32_t padding_cols_right = pads.at(3);

    uint32_t channels = this->Channels();
    CHECK_GT(channels, 0);

    uint32_t old_rows = this->Rows();
    uint32_t old_cols = this->Cols();
    uint32_t new_rows = old_rows + padding_rows_up + padding_rows_bottom;
    uint32_t new_cols = old_cols + padding_cols_left + padding_cols_right;

    arma::fcube padded_cube(new_rows, new_cols, channels);
    padded_cube.fill(padding_value);

    for (uint32_t i = 0; i < channels; ++i) {
        const arma::fmat& sub_mat = this->data_.slice(i);
        arma::fmat& padded_mat = padded_cube.slice(i);
        padded_mat.submat(
            padding_rows_up, 
            padding_cols_left, 
            padding_rows_up + old_rows - 1,
            padding_cols_left + old_cols - 1
        ) = sub_mat;
    }

    this->data_ = std::move(padded_cube);
    if (this->shape_.size() == 1) {
        this->shape_ = std::vector<uint32_t>{new_cols};
    } else if (this->shape_.size() == 2) {
        this->shape_ = std::vector<uint32_t>{new_rows, new_cols};
    } else {
        this->shape_ = std::vector<uint32_t>{channels, new_rows, new_cols};
    }
}

void Tensor<float>::Flatten(bool row_major) {
    CHECK(!this->data_.empty());

    uint32_t total_size = this->Size();
    if (row_major) {
        std::vector<float> row_major_values = this->Values(true);
        this->data_.reshape(1, total_size, 1);
        this->shape_ = std::vector<uint32_t>{total_size};
        std::copy(row_major_values.begin(), row_major_values.end(), this->data_.memptr());
    } else {
        this->data_.reshape(1, total_size, 1);
        this->shape_ = std::vector<uint32_t>{total_size};
    }
}

void Tensor<float>::Show() {
    for (uint32_t i = 0; i < this->Channels(); ++i) {
        LOG(INFO) << "Channel: " << i;
        LOG(INFO) << "\n" << this->data_.slice(i);
    }
}

}
