#include "layer.h"
#include <cmath>

// Constructor
Layer::Layer(int M, int N, int O) {
    this->M = M;
    this->N = N;
    this->O = O;

    output = new float[O]();
    preact = new float[O]();
    bias = new float[N]();
    weight = new float[M * N]();

    for (int i = 0; i < N; ++i) {
        bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
        for (int j = 0; j < M; ++j) {
            weight[i * M + j] = 0.5f - float(rand()) / float(RAND_MAX);
        }
    }
}

// Destructor
Layer::~Layer() {
    delete[] output;
    delete[] preact;
    delete[] bias;
    delete[] weight;
}

void Layer::setOutput(float *data) {
    for (int i = 0; i < O; ++i) {
        output[i] = data[i];
    }
}

void Layer::clear() {
    for (int i = 0; i < O; ++i) {
        output[i] = 0;
        preact[i] = 0;
    }
}

float step_function(float v) {
    return 1 / (1 + exp(-v));
}

void apply_step_function(float *input, float *output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = step_function(input[i]);
    }
}

void makeError(float *err, float *output, unsigned int Y, int N) {
    for (int i = 0; i < N; ++i) {
        err[i] = ((Y == i ? 1.0f : 0.0f) - output[i]);
    }
}

void apply_grad(float *output, float *grad, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] += dt * grad[i];
    }
}

void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]) {
    for (int i3 = 0; i3 < 6; ++i3) {
        for (int i4 = 0; i4 < 24; ++i4) {
            for (int i5 = 0; i5 < 24; ++i5) {
                for (int i1 = 0; i1 < 5; ++i1) {
                    for (int i2 = 0; i2 < 5; ++i2) {
                        preact[i3][i4][i5] += weight[i3][i1][i2] * input[i4 + i1][i5 + i2];
                    }
                }
            }
        }
    }
}

void fp_bias_c1(float preact[6][24][24], float bias[6]) {
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 24; ++i2) {
            for (int i3 = 0; i3 < 24; ++i3) {
                preact[i1][i2][i3] += bias[i1];
            }
        }
    }
}
