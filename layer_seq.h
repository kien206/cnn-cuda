#ifndef LAYER_H
#define LAYER_H

#include <cstdlib> // For rand()
const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
public:
    int M; // Number of inputs to a layer
    int N; // Number of neurons in a layer
    int O; // Number of outputs from a layer

    float *output;  // Output values of this layer
    float *preact;  // Preactivation values (before activation function is applied)
    float *bias;    // Biases associated with each neuron in the layer
    float *weight;  // Weights of the connections

    // Constructor and Destructor
    Layer(int M, int N, int O);
    ~Layer();

    // Methods
    void setOutput(float *data);
    void clear();
    void apply_grad(float *output, float *grad, int N);
    void makeError(float *err, float *output, unsigned int Y, int N);
    void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
    void fp_bias_c1(float preact[6][24][24], float bias[6]);
    void apply_step_function(float *input, float *output, int N)
private:
    static float step_function(float v);
    static void apply_step_function(float *input, float *output, int N);
};

#endif // LAYER_H
