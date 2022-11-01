#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "mnist_helper.h"
#include <stdio.h>

#define N_NEURONS_LI 784
#define N_NEURONS_L1 300
#define N_NEURONS_L2 100
#define N_NEURONS_L3 100
#define N_NEURONS_LO 10

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

// Weight struct for SGD
typedef struct weight_struct_t {
    double w; // weight
    double dw; // change in w to be applied
    double v;
    double mean; // mean
    double variance; // variance
} weight_struct_t;

// Weight matrices
extern weight_struct_t w_LI_L1[N_NEURONS_LI][N_NEURONS_L1]; // defined in neural_network.c
extern weight_struct_t w_L1_L2[N_NEURONS_L1][N_NEURONS_L2]; // defined in neural_network.c
extern weight_struct_t w_L2_L3[N_NEURONS_L2][N_NEURONS_L3]; // defined in neural_network.c
extern weight_struct_t w_L3_LO[N_NEURONS_L3][N_NEURONS_LO]; // defined in neural_network.c

// Grdaients for optimiser
extern double dL_dW_L3_LO[1][N_NEURONS_L3*N_NEURONS_LO]; // defined in neural_network.c
extern double dL_dW_L2_L3[1][N_NEURONS_L2*N_NEURONS_L3]; // defined in neural_network.c
extern double dL_dW_L1_L2[1][N_NEURONS_L1*N_NEURONS_L2]; // defined in neural_network.c
extern double dL_dW_LI_L1[1][N_NEURONS_LI*N_NEURONS_L1]; // defined in neural_network.c

// Functions required in main.c and optimiser.c
void initialise_nn(void);
void evaluate_forward_pass(uint8_t** dataset, int n);
double compute_xent_loss(uint8_t correct_label);
void evaluate_backward_pass(uint8_t label, unsigned int input_class_index);
void evaluate_backward_pass_sparse(uint8_t label, unsigned int input_class_index);
void evaluate_weight_updates(void);
double evaluate_testing_accuracy(void);

#endif /* NEURAL_NETWORK */
