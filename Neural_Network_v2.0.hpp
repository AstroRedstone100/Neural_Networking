#pragma once

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <cmath>

static std::mt19937 gen(std::random_device{}());

#define CEL 0
#define MSE 1

struct matrix {
    std::vector<float> data;
    int rows, cols, size;

    matrix() {}

    matrix(int Rows, int Cols, int Size, bool random)
        : rows(Rows), cols(Cols), size(Size), data(Size) {
            if (random) {
                const unsigned int range = std::sqrt(6.0f / Rows); /* The initialization range for weights in networks with ReLU activations,
                                                                      suggested by He (Kaiming) initialization to avoid learning problems */
                std::uniform_real_distribution<float> dist(-range, range);
                for (auto& element : data) {element = dist(gen);}
            }
        }

    matrix& operator=(const matrix& other) {
        if (this != &other) {
            this->data= other.data; this->size = other.size;
            this->rows = other.rows; this->cols = other.cols;
        }

        return *this;
    }
};

inline matrix operator+(const matrix& a, const matrix& b) {
    const int rows = a.rows, cols = a.cols, size = a.size;
    matrix sum(rows, cols, size, false);
    if ((b.rows != rows) || (b.cols != cols)) {
        std::cerr << "\nIncompatible dimensions for addition"; return sum;
    } for (int row_ = 0; row_ < size; row_ += cols) {
        for (int x = 0; x < cols; x++) {
            const int index = row_ + x;
            sum.data[index] = a.data[index] + b.data[index];
        }
    } return sum;
}

inline matrix operator-(const matrix& a, const matrix& b) {
    const int rows = a.rows, cols = a.cols, size = a.size;
    matrix diff(rows, cols, size, false);
    if ((b.rows != rows) || (b.cols != cols)) {
        std::cerr << "\nIncompatible dimensions for subtraction"; return diff;
    } for (int row_ = 0; row_ < size; row_ += cols) {
        for (int x = 0; x < cols; x++) {
            const int index = row_ + x;
            diff.data[index] = a.data[index] - b.data[index];
        }
    } return diff;
}

inline void operator-=(matrix& a, const matrix& b) {a = a - b;}

inline matrix operator*(const float& s, const matrix& m) {
    const int size = m.size;
    matrix resultant(m.rows, m.cols, size, false);
    for (int index = 0; index < size; index++) {resultant.data[index] = m.data[index] * s;}
    return resultant;
}

inline matrix transpose(const matrix& m) {
    const int size = m.size, rows = m.rows, cols = m.cols;
    matrix result(cols, rows, size, false);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {result.data[x * rows + y] = m.data[y * cols + x];}
    } return result;
}

inline matrix operator*(const matrix& a, const matrix& b) {
    const int rows = a.rows, cols = b.cols, dimension1 = a.cols, dimension2 = b.rows;
    matrix product(rows, cols, rows * cols, false);
    if (dimension1 != dimension2) {
        std::cerr << "\nIncompatible dimensions for multiplication"; return product;
    } const matrix new_mat = transpose(b);
    for (int row1 = 0; row1 < rows; row1++) {
        for (int row2 = 0; row2 < cols; row2++) {
            float sum = 0.0f;
            for (int idx = 0; idx < dimension1; idx++) {sum += a.data[row1 * dimension1 + idx] * b.data[row2 * dimension2 + idx];}
            product.data[row1 * cols + row2] = sum;
        }
    } return product;
}

inline matrix hadamard_product(const matrix& a, const matrix& b) {
    const int rows = a.rows, cols = a.cols, size = a.size;
    matrix product(rows, cols, size, false);
    if ((b.rows != rows) || (b.cols != cols)) {
        std::cerr << "\nIncompatible dimensions for hadamard product"; return product;
    } for (int row_ = 0; row_ < size; row_ += cols) {
        for (int x = 0; x < cols; x++) {
            const int index = row_ + x;
            product.data[index] = a.data[index] * b.data[index];
        }
    } return product;
}

inline int relu(int x) {return ((x > 0) ? x : 0);}

inline matrix ReLU(const matrix& m, const int& size) {
    matrix n(m.rows, m.cols, size, false);
    for (int index = 0; index < size; index++) {n.data[index] = relu(m.data[index]);}
    return n;
}

inline matrix ReLU_(const matrix& m) {
    const int size = m.size;
    matrix resultant(m.rows, m.cols, size, false);
    for (int idx = 0; idx < size; idx++) {resultant.data[idx] = (m.data[idx] > 0);}
    return resultant;
}

inline matrix softmax(const matrix& m, const int& size, const int& cols) {
    matrix resultant(m.rows, cols, size, false);
    for (int jx = 0; jx < size; jx += cols) {
        float denominator = 0.0f;
        for (int i = 0; i < cols; i++) {denominator += std::exp(m.data[jx + i]);}
        for (int i = 0; i < cols; i++) {
            const int index = jx + i;
            resultant.data[index] = std::exp(m.data[index]) / denominator;
        }
    } return resultant;
}

struct neural_network {
    std::vector<matrix> weights, weight_grads, biases, bias_grads, Z, A;
    int layers1, layers2, layers3; float N1, learning_rate;

    neural_network(std::vector<int> neuron_distri, int layers_1, int batch, float lr)
        : weights(layers_1), weight_grads(layers_1), biases(layers_1), bias_grads(layers_1), Z(layers_1), A(layers_1 + 1), learning_rate(lr) {
        layers1 = layers_1; layers2 = layers1 - 1; layers3 = layers2 - 1; N1 = 1.0f / batch;
        for (int count = 0; count < layers_1; count++) {
            const int inputs = neuron_distri[count], outputs = neuron_distri[count+1];
            weights[count] = matrix(inputs, outputs, inputs * outputs, true);
            biases[count] = matrix(batch, outputs, batch * outputs, false);
        }
    }

    matrix forward(matrix& x, const int& loss_type) {
        A[0] = x;
        for (int layer = 0; layer < layers1; layer++) {
            const matrix z = x * weights[layer] + biases[layer];
            if (layer != layers2) {x = ReLU(z, z.size);}
            else if (loss_type == CEL) {x = softmax(z, z.size, z.cols);}
            Z[layer] = z; A[layer + 1] = x;
        } return x;
    }

    void backward_pass(const matrix& actual) {
        matrix del = N1 * (Z[layers2] - actual);
        weight_grads[layers2] = transpose(A[layers2]) * del; bias_grads[layers2] = del;
        for (int layer = layers3; layer >= 0; layer--) {
            del = hadamard_product((del * transpose(weights[layer + 1])), ReLU_(Z[layer]));
            weight_grads[layer] = transpose(A[layer]) * del; bias_grads[layer] = del;
        }
    }

    void optimize() {
        for (int layer = 0; layer < layers1; layer++) {
            weights[layer] -= learning_rate * weight_grads[layer];
            biases[layer] -= learning_rate * bias_grads[layer];
        }
    }
};