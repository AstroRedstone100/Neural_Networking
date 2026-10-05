#include <cmath>
#include <vector>
#include <random>
#include <iostream>

static std::mt19937 gen(std::random_device{}()); // The Mersenne Twister generator
static std::uniform_real_distribution<float> dist(-10.0f, 10.0f); // The distributor

struct matrix {
    std::vector<float> values; // [row][column]
    int rows, cols;

    matrix(int rownum, int colnum, float value) :  values(rownum*colnum, value), rows(rownum), cols(colnum) {}

    matrix(int rownum, int colnum) : values(rownum*colnum), rows(rownum), cols(colnum) {
        const int total = values.size();
        for (int idx = 0; idx < total; idx++) {values[idx] = dist(gen);}
    }

    matrix() {}
};

inline matrix operator+(const matrix& a, const matrix& b) {
    matrix ans = a; unsigned int size = a.values.size();
    for (unsigned int idx = 0; idx < size; idx++) {ans.values[idx] += b.values[idx];}
    return ans;
}

inline matrix operator-=(matrix& a, const matrix& b) {
    const unsigned int size = a.values.size();
    for (unsigned int idx = 0; idx < size; idx++) {a.values[idx] -= b.values[idx];}
    return a;
}

inline matrix operator-(const matrix& a, const std::vector<unsigned int>& b) { // A custom matrix subtraction operator which subtracts 1 from
    matrix ans = a; const unsigned int size = b.size(), w = a.cols;            // those values of 'a' whose coordinates are stored as b[x] = y
    for (unsigned int idx = 0; idx < size; idx++) {ans.values[idx + b[idx] * w] -= 1.0f;}
    return ans;
}

inline matrix operator*(const matrix& a, const matrix& b) { // Matrix * Matrix
    if (a.cols != b.rows) {std::cerr << "\nError 101: Dimensional Mismatch"; return a;}
    matrix answer(a.rows, b.cols, 0.0f);
    const unsigned int mx = b.cols, my = a.rows, nx = a.cols;
    for (unsigned int y_a = 0; y_a < my; y_a++) {
        for (unsigned int x_b = 0; x_b < mx; x_b++) {
            int temp = 0.0f; const int k1 = y_a * nx;
            for (unsigned int idx = 0; idx < nx; idx++) {temp += a.values[k1+idx] * b.values[idx*mx+x_b];}
            answer.values[y_a * mx + x_b] = temp;
        }
    } return answer;
}

inline matrix operator*(matrix& a, const float& m) { // Matrix * Scalar
    for (auto& val : a.values) {val = val * m;}
    return a;
}

inline matrix operator/(const matrix& a, const float& m) {
    matrix answer = a;
    for (auto& val : answer.values) {val = val / m;}
    return answer;
}

matrix transpose(const matrix& a) {
    const unsigned int w = a.cols, h = a.rows;
    matrix ans(w, h, 0.0f);
    for (unsigned int y = 0; y < h; y++) {
        for (unsigned int x = 0; x < w; x++) {
            ans.values[y + x * h] = a.values[x + y * w];
        }
    } return ans;
}

matrix hadamard(const matrix& a, const std::vector<unsigned int>& b) { // A custom hadamard product function designed for evaluating the
    matrix mat(a.rows, a.cols, 0.0f);                                  // hadamard product of a normal one and a binary one
    for (auto const& idx : b) {mat.values[idx] = a.values[idx];}
    return mat;
}

float relu(const float& a) {return ((a > 0.0f) ? a : 0.0f);}

void ReLU(matrix& m) {for (auto& value : m.values) {value = relu(value);}}
std::vector<unsigned int> ReLU_der(const matrix& m) { // The matrix ReLU derivative function
    std::vector<unsigned int> vec; const unsigned int size = m.values.size();
    for (unsigned int idx = 0; idx < size; idx++) {if (m.values[idx] > 0.0f) {vec.push_back(idx);}}
    return vec;
}

matrix softmax(const matrix& m) {
    const int mx = m.cols, my = m.rows * mx;
    matrix mat = m;
    for (int x = 0; x < mx; x++) {
        float temp = 0.0f; const int mY = my + x;
        for (int y = x; y < mY; y += mx) {temp += std::exp(m.values[y]);}
        for (int y = x; y < mY; y += mx) {mat.values[y] = std::exp(m.values[y]) / temp;}
    } return mat;
}

enum optimizer_type {SGD};

class nn {
public:
    
    struct classifier {
        std::vector<matrix> biases, bias_grads;
        std::vector<matrix> weights, weight_grads; /* w11 w21 w31 ...
                                                      w12 w22 w32 ..., x -> input_in, y -> output_in */
        int size, pred_size, pre_size, batch_size; // The size of the 'distr' vector, its predecessors and the batch_size of foward_passing
        float learning_rate;
        bool train;

        std::vector<std::vector<unsigned int>> prev_inps; // This stores the history of inp
        std::vector<matrix> prev_relus;

        classifier(const std::vector<unsigned int> distr, const unsigned int Batch) {
            batch_size = Batch;
            size = distr.size(); pred_size = size - 1, pre_size = size - 2;
            for (unsigned int idx = 1; idx < size; idx++) {
                biases.push_back(matrix(distr[idx], batch_size, 0.0f));
                weights.push_back(matrix(distr[idx], distr[idx-1]));
            } bias_grads = biases; weight_grads = weights;
            
            train = false;
            prev_inps = std::vector<std::vector<unsigned int>>(pred_size);
            prev_relus = std::vector<matrix>(pred_size);
            learning_rate = 0.01f;
        }

        matrix forward(const matrix input) {
            matrix inp = input; if (train) {prev_relus[0] = input;}
            for (unsigned int layer = 0; layer < pred_size; layer++) {
                inp = weights[layer] * inp + biases[layer];
                if (train && layer < pre_size) {prev_inps[layer+1] = ReLU_der(inp);} // prev_inps[0] is skipped because it doesn't really appear
                if (layer < pre_size) {ReLU(inp); if (train) {prev_relus[layer+1] = inp;}}
            } return inp;
        }

        void backpropagate(matrix& outp, const std::vector<unsigned int>& target) {
            const matrix y_hat = softmax(outp);
            matrix dL = (y_hat - target) / batch_size;

            for (int layer = pre_size; layer >= 0; layer--) {
                weight_grads[layer] = dL * transpose(prev_relus[layer]);
                bias_grads[layer] = dL;
                dL = hadamard(transpose(weights[layer]) * dL, prev_inps[layer]);
            }
        }

        void optimize(const optimizer_type type) {
            if (type == SGD) {
                for (unsigned int idx = 0; idx < pred_size; idx++) {
                    weights[idx] -= weight_grads[idx] * learning_rate;
                    biases[idx] -= bias_grads[idx] * learning_rate;
                }
            }
        }
    };
};

int main() {
    nn::classifier neural_net({784, 128, 10}, 32);
    const matrix Input(784, 32);

    const matrix out = neural_net.forward(Input);
    
    return 0;
}