#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cstdio>
#include <ctime>
#include <cmath>

// Global declarations
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input(0, 0, 28 * 28);
static Layer l_c1(5 * 5, 6, 24 * 24 * 6);
static Layer l_s1(4 * 4, 1, 6 * 6 * 6);
static Layer l_f(6 * 6 * 6, 10, 10);

// Function declarations
static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

// Function to load MNIST data
static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

static double forward_pass(double data[28][28]) {
    float input[28][28];

    for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

    clock_t start, end;
	start = clock();

	l_input.setOutput((float *)input);

    fp_preact_c1((float ()[28])l_input.output, (float ()[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
    fp_bias_c1((float (*)[24][24])l_c1.preact, l_c1.bias);
    apply_step_function(l_c1.preact, l_c1.output, l_c1.O);

    fp_preact_s1((float ()[24][24])l_c1.output, (float ()[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
    fp_bias_s1((float (*)[6][6])l_s1.preact, l_s1.bias);
	apply_step_function(l_s1.preact, l_s1.output, l_s1.O);

	fp_preact_f((float ()[6][6])l_s1.output, l_f.preact, (float ()[6][6][6])l_f.weight);
	fp_bias_f(l_f.preact, l_f.bias);
	apply_step_function(l_f.preact, l_f.output, l_f.O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static double back_pass() {
    clock_t start, end;

	start = clock();
    
    bp_weight_f((float ()[6][6][6])l_f.d_weight, l_f.d_preact, (float ()[6][6])l_s1.output);
	bp_bias_f(l_f.bias, l_f.d_preact);

	bp_output_s1((float ()[6][6])l_s1.d_output, (float ()[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s1((float ()[6][6])l_s1.d_preact, (float ()[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
	bp_weight_s1((float ()[4][4])l_s1.d_weight, (float ()[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
	bp_bias_s1(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

	bp_output_c1((float ()[24][24])l_c1.d_output, (float ()[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
	bp_preact_c1((float ()[24][24])l_c1.d_preact, (float ()[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	bp_weight_c1((float ()[5][5])l_c1.d_weight, (float ()[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
	bp_bias_c1(l_c1.bias, (float (*)[24][24])l_c1.d_preact);


	apply_grad(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

int main(int argc, char **argv) {
    srand(static_cast<unsigned int>(time(nullptr)));

    loaddata();
    learn();
    test();

    return 0;
}

void learn() {
    printf("Learning\n");
    for (int i = 0; i < train_cnt; ++i) {
        forward_pass(train_set[i].data);  // Forward propagation
        back_pass();  // Backward propagation
    }
}

unsigned int classify(double data[28][28]) {
    forward_pass(data);
    unsigned int max = 0;

    for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

void test() {
    int error_count = 0;
    for (int i = 0; i < test_cnt; ++i) {
        if (classify(test_set[i].data) != test_set[i].label) {
            ++error_count;
        }
    }
    printf("Error Rate: %.2lf%%\n", double(error_count) / double(test_cnt) * 100.0);
}
