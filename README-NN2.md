# Neural Network version-2.0
This is one of my biggest, smoothest and best projects ever because it literally combines elements of all three. The below sections explain about the implementation:

## Overall Implementation
I use a structure inspired by that of PyTorch. Similar to it, I have a `struct` for the neural network (though PyTorch uses a `class`) and I also have custom `matrix` structs containing dimensional, size and data properties of the matrix. I also relied on a single `std::vector` containing the data of the matrices because nested vectors introduce unnecessary extra overhead and allocation, slowing down the program. I only included two types of loss functions because I myself was going to use only two. Anyways, since the overall implementation is explained, let us focus on the specific parts of the program:

## Particular Implementations
Let us start off from the constructor/initialization part:

### Constructor
```cpp
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
```
I don't do much here, i.e., just initialize the sizes and the number of weights and biases and other vectors. For those wondering what the layers parameters are for, they are for storing the predecessors of `layers_1` where, `layers_1` is the effective layer count, i.e., original layer count's predecessor as the first input layer doesn't have any parameters.

### Forward Pass
```cpp
   matrix forward(matrix& x, const int& loss_type) {
        A[0] = x;
        for (int layer = 0; layer < layers1; layer++) {
            const matrix z = x * weights[layer] + biases[layer];
            if (layer != layers2) {x = ReLU(z, z.size);}
            else if (loss_type == CEL) {x = softmax(z, z.size, z.cols);}
            Z[layer] = z; A[layer + 1] = x;
        } return x;
    }
```
You can see that `A[0]` is initialized with `x` because later in the code in back propagation, `A[0]` is utilized and according to our logic flow where z = aw + b, for the first layer, a = `x` causing `A[0]` to be `x`. You can also see that the final `softmax()` activation is only applied for Cross-Entropy Loss because RL networks and Regression networks utilize raw final scores rather than the probabilities provided by `softmax()`.

### Backward Pass
```cpp
   void backward_pass(const matrix& actual) {
        matrix del = N1 * (Z[layers2] - actual);
        weight_grads[layers2] = transpose(A[layers2]) * del; bias_grads[layers2] = del;
        for (int layer = layers3; layer >= 0; layer--) {
            del = hadamard_product((del * transpose(weights[layer + 1])), ReLU_(Z[layer]));
            weight_grads[layer] = transpose(A[layer]) * del; bias_grads[layer] = del;
        }
    }
```
You can see that I start off by manually calculating the first `del` (dL/dz) because the first one doesn't depend on any other `del` to compute itself. Then, using the formulae for backpropagation and gradient calculation, I calculate further dels and the other gradients and store that in my grad vectors.

### Optimizing
```cpp
   void optimize() {
        for (int layer = 0; layer < layers1; layer++) {
            weights[layer] -= learning_rate * weight_grads[layer];
            biases[layer] -= learning_rate * bias_grads[layer];
        }
    }
```
There is nothing much to explain in this section because we're just looping through the layers and updating the parameters.

## Conclusion
This was a fun project. Fun fact: I coded it in 1 day (though including the research, derivation and intuition process increases the overall time taken to 3-4 days). And, don't criticise me for using 3 separate ints just to store the `layers_1` predecessors! It was also fun using the formatting tools of markdown for the first time.
