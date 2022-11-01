#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "neural_network.h"

// ******** Neuron type structs ********
typedef struct n_ReLu {
    double z; // weighted-sum of input
    double h; // output activation
} n_ReLu;

typedef struct n_Out {
    double z; // weighted-sum of input
    double P; // probability
} n_Out;

// ******** Neuron layers ********
n_ReLu n_L1[N_NEURONS_L1];
n_ReLu n_L2[N_NEURONS_L2];
n_ReLu n_L3[N_NEURONS_L3];
n_Out n_LO[N_NEURONS_LO];

// ******** Weight matrices ********
weight_struct_t w_LI_L1[N_NEURONS_LI][N_NEURONS_L1];
weight_struct_t w_L1_L2[N_NEURONS_L1][N_NEURONS_L2];
weight_struct_t w_L2_L3[N_NEURONS_L2][N_NEURONS_L3];
weight_struct_t w_L3_LO[N_NEURONS_L3][N_NEURONS_LO];

// ******** Gradients for Optimiser ********
double dL_dW_L3_LO[1][N_NEURONS_L3*N_NEURONS_LO];
double dL_dW_L2_L3[1][N_NEURONS_L2*N_NEURONS_L3];
double dL_dW_L1_L2[1][N_NEURONS_L1*N_NEURONS_L2];
double dL_dW_LI_L1[1][N_NEURONS_LI*N_NEURONS_L1];

// ******** Jacobians (internal to neural_network.c) ********
double dL_dP[1][N_NEURONS_LO];
double dP_dhO[N_NEURONS_LO][N_NEURONS_LO];
double dhO_dW_L3_LO[N_NEURONS_LO][N_NEURONS_L3*N_NEURONS_LO];
double dhO_dh3[N_NEURONS_LO][N_NEURONS_L3];
double dh3_dW_L2_L3[N_NEURONS_L3][N_NEURONS_L2*N_NEURONS_L3];
double dh3_dh2[N_NEURONS_L3][N_NEURONS_L2];
double dh2_dW_L1_L2[N_NEURONS_L2][N_NEURONS_L1*N_NEURONS_L2];
double dh2_dh1[N_NEURONS_L2][N_NEURONS_L1];
double dh1_dW_LI_L1[N_NEURONS_L1][N_NEURONS_LI*N_NEURONS_L1];

// ******** Intermediate gradients (internal to neural network.c - avoids re-evaluation during backward pass) ********
double dL_dhO[1][N_NEURONS_LO];
double dL_dh3[1][N_NEURONS_L3];
double dL_dh2[1][N_NEURONS_L2];
double dL_dh1[1][N_NEURONS_L1];

// ******** Internal function declarations ********
// General functions
double drand(double min, double max);
void print_matrix(unsigned int rows, unsigned int cols, double matrix[][cols]);
void inline double_matrix_multiply(unsigned int rows1, unsigned int cols1, double mat1[][cols1],
                            unsigned int rows2, unsigned int cols2, double mat2[][cols2],
                            unsigned int rows_out, unsigned int cols_out, double mat_out[][cols_out]);

// Initialisation functions
void initialise_neurons(void);
void initialise_weight_matrices(void);
void initialise_gradients_and_jacobians(void);

// Foward pass
void compute_softmax(void);

//Backward pass
void evaluate_dL_dP(uint8_t label);
void evaluate_dP_dhO(void);
void inline evaluate_dh_dW_Lh_Lprev(unsigned int n_h, unsigned int n_prev, n_ReLu neur[n_h], unsigned int n_W, double J[][n_W]);
void inline evaluate_dh_dh_prev_inc_pre_act(unsigned int n_h, unsigned int n_prev, n_ReLu n_Lh[n_h], weight_struct_t W[][n_h], double J[][n_prev]);
void inline evaluate_dh_dW_Lh_Lprev_sparse(unsigned int n_h, unsigned int n_prev, n_ReLu neur[n_h], unsigned int n_W, double J[][n_W]);
void inline sparse_multiply_dL_dh_by_dh_dW_LH_LH_prev(unsigned int rows1, unsigned int cols1, double mat1[][cols1], // dl_dh
                                               unsigned int rows2, unsigned int cols2, double mat2[][cols2], // dh_dW_LH_LH_prev
                                               unsigned int rows_out, unsigned int cols_out, double mat_out[][cols_out]);

// ******** Function definitions ********
double drand(double min, double max){ //
    double random_double = 0.0;
    random_double = (float) rand() / RAND_MAX; // note rand() only generates a 32-bit random number (no use casting to double) - could use drand48() to generate doubles
    random_double = (random_double * (max - min)) + min;
    return random_double;
}

void print_matrix(unsigned int rows, unsigned int cols, double matrix[][cols]){
    for (int i=0; i< rows; i++){
        for (int j=0; j< cols; j++){
            printf("%0.6f    ", matrix[i][j]);
        }
        printf("\n\n");
    }
}

void inline double_matrix_multiply(unsigned int rows1, unsigned int cols1, double mat1[][cols1],
                            unsigned int rows2, unsigned int cols2, double mat2[][cols2],
                            unsigned int rows_out, unsigned int cols_out, double mat_out[][cols_out]){
    
    if (cols1 != rows2){
        printf("Incorrect matrix format (cols1 != rows2");
        exit(1);
    }
    if (rows1 != rows_out){
        printf("Incorrect matrix format (rows1 != rows_out)");
        exit(1);
    }
    if (cols2 != cols_out){
        printf("Incorrect matrix format (cols2 != cols_out)");
        exit(1);
    }
    
    for (int i = 0; i < rows1; i++){
        for (int j = 0; j < cols2; j++){
            mat_out[i][j] = 0.0;
            for (int k = 0; k < cols1; k++){
                mat_out[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    
    // Example data structure to test matrix multiply
    //    double A[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    //    double B[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    //    double C[2][3] = {{0, 0, 0}, {0, 0, 0}};
    //
    //
    //    double_matrix_multiply(2, 4, A, 4, 3, B, 2, 3, C);
    //    print_matrix(2, 3, C);
    
}

void initialise_neurons(void){
    printf("Initialising neuron datastructures\n\n");
    for (int i=0; i<N_NEURONS_L1; i++){
        n_L1[i].h = 0.0;
        n_L1[i].z = 0.0;
    }
    for (int i=0; i<N_NEURONS_L2; i++){
        n_L2[i].h = 0.0;
        n_L2[i].z = 0.0;
    }
    for (int i=0; i<N_NEURONS_L3; i++){
        n_L3[i].h = 0.0;
        n_L3[i].z = 0.0;
    }
    for (int i=0; i<N_NEURONS_LO; i++){
        n_LO[i].z = 0.0;
        n_LO[i].P = 0.0;
    }
}

void initialise_weight_matrices(void){
    printf("Initialising weight matrices\n");
    
    double w_limit = sqrt(6) / sqrt(N_NEURONS_LI + N_NEURONS_L1);
    // Input -> L1
    for (int i=0; i < N_NEURONS_LI; i++){
        for (int j=0; j <N_NEURONS_L1; j++){
            w_LI_L1[i][j].w = drand(-w_limit, w_limit);
            w_LI_L1[i][j].dw = 0.0;
            w_LI_L1[i][j].v = 0.0;

            w_LI_L1[i][j].mean = 0.0;
            w_LI_L1[i][j].variance = 0.0;
        }
    }
    
    // L1 -> L2
    w_limit = sqrt(6) / sqrt(N_NEURONS_L1 + N_NEURONS_L2);
    for (int i=0; i < N_NEURONS_L1; i++){
        for (int j=0; j <N_NEURONS_L2; j++){
            w_L1_L2[i][j].w = drand(-w_limit, w_limit);
            w_L1_L2[i][j].dw = 0.0;
            w_L1_L2[i][j].v = 0.0;

            w_L1_L2[i][j].mean = 0.0;
            w_L1_L2[i][j].variance = 0.0;
        }
    }
    
    // L2->L3
    w_limit = sqrt(6) / sqrt(N_NEURONS_L2+N_NEURONS_L3);
    for (int i=0; i < N_NEURONS_L2; i++){
        for (int j=0; j <N_NEURONS_L3; j++){
            w_L2_L3[i][j].w = drand(-w_limit, w_limit);
            w_L2_L3[i][j].dw = 0.0;
            w_L2_L3[i][j].v = 0.0;

            w_L2_L3[i][j].mean = 0.0;
            w_L2_L3[i][j].variance = 0.0;
        }
    }
    
    // L3-Output
    w_limit = sqrt(6) / sqrt(N_NEURONS_L3+N_NEURONS_LO);
    for (int i=0; i < N_NEURONS_L3; i++){
        for (int j=0; j <N_NEURONS_LO; j++){
            w_L3_LO[i][j].w = drand(-w_limit, w_limit);
            w_L3_LO[i][j].dw = 0.0;
            w_L3_LO[i][j].v = 0.0;

            w_L3_LO[i][j].mean = 0.0;
            w_L3_LO[i][j].variance = 0.0;
        }
    }
}

void initialise_gradients_and_jacobians(void){
    printf("Initialising gradient and Jacobian datastructures\n");
    
    // ******** Gradients for Optimiser ********
    for (int i=0; i<N_NEURONS_L3*N_NEURONS_LO; i++){
        dL_dW_L3_LO[0][i] = 0.0;
    }
    for (int i=0; i< N_NEURONS_L2*N_NEURONS_L3; i++){
        dL_dW_L2_L3[0][i] = 0.0;
    }
    
    for (int i = 0; i < N_NEURONS_L1*N_NEURONS_L2; i++){
        dL_dW_L1_L2[0][i] = 0.0;
    }
    
    for (int i=0; i < N_NEURONS_LI*N_NEURONS_L1; i++){
        dL_dW_LI_L1[0][i] = 0.0;
    }
    
    // ******** Jacobians (internal to neural_network.c) ********
    for (int i=0; i< N_NEURONS_LO; i++){
        dL_dP[0][i] = 0.0;
    }
    for (int i=0; i< N_NEURONS_LO;i++){
        for(int j= 0; j < N_NEURONS_LO; j++){
            dP_dhO[i][j] = 0.0;
        }
    }
    for (int i=0; i< N_NEURONS_LO;i++){
        for(int j= 0; j < N_NEURONS_L3*N_NEURONS_LO; j++){
            dhO_dW_L3_LO[i][j] = 0.0;;
        }
    }
    for (int i=0; i< N_NEURONS_LO;i++){
        for(int j= 0; j < N_NEURONS_L3; j++){
            dhO_dh3[i][j] = 0.0;;
        }
    }
    for (int i=0; i< N_NEURONS_L3;i++){
        for(int j= 0; j < N_NEURONS_L2*N_NEURONS_L3; j++){
            dh3_dW_L2_L3[i][j] = 0.0;
        }
    }
    for (int i=0; i< N_NEURONS_L3;i++){
        for(int j= 0; j < N_NEURONS_L2; j++){
            dh3_dh2[i][j] = 0.0;
        }
    }
    for (int i=0; i < N_NEURONS_L2;i++){
        for(int j= 0; j < N_NEURONS_L1*N_NEURONS_L2; j++){
            dh2_dW_L1_L2[i][j] = 0.0;
        }
    }
    for (int i=0; i < N_NEURONS_L2;i++){
        for(int j= 0; j < N_NEURONS_L1; j++){
            dh2_dh1[i][j] = 0.0;
        }
    }
    for (int i=0; i< N_NEURONS_L1;i++){
        for(int j= 0; j < N_NEURONS_LI*N_NEURONS_L1; j++){
            dh1_dW_LI_L1[i][j] = 0.0;
        }
    }
    
    // Intermediate gradients
    for (int i = 0; i<N_NEURONS_LO; i++){
        dL_dhO[0][i] = 0.0;
    }
    for (int i = 0; i<N_NEURONS_L3; i++){
        dL_dh3[0][i] = 0.0;
    }
    for (int i = 0; i < N_NEURONS_L2; i++){
        dL_dh2[0][i] = 0.0;
    }
    for (int i=0; i< N_NEURONS_L1; i++){
        dL_dh1[0][i] = 0.0;
    }
}

// Backward pass functions
void evaluate_dL_dP(uint8_t label){
    // reset Jacbobian from last evaluation
    for (int j = 0; j < N_NEURONS_LO; j++){
        dL_dP[0][j] = 0.0;
    }
    dL_dP[0][label] = -1/(n_LO[label].P);
}

void evaluate_dP_dhO(void){
    for (int i = 0; i < N_NEURONS_LO; i++){
        for (int j = 0; j<N_NEURONS_LO; j++){
            // No need to reset term to zero, as every term in Jacobian is to be re-evaluated
            if(i == j){
                dP_dhO[i][j] = n_LO[i].P * (1 - n_LO[j].P);
            } else {
                dP_dhO[i][j] = - n_LO[i].P * n_LO[j].P;
            }
        }
    }
}

void inline evaluate_dh_dW_Lh_Lprev(unsigned int n_h,
                             unsigned int n_prev,
                             n_ReLu neur[n_prev],
                             unsigned int n_W, double J[][n_W]){
    for (int i = 0; i < n_h; i++){
        for (int j = 0; j < (n_W); j++){
            if (j >= (i * n_prev) && j < ((i+1) * n_prev)) {
                J[i][j] = neur[j - (i * n_prev)].h;
            } else {
                J[i][j] = 0.0;
            }
        }
    }
}

void inline evaluate_dh_dW_Lh_Lprev_sparse(unsigned int n_h,
                                    unsigned int n_prev,
                                    n_ReLu neur[n_prev],
                                    unsigned int n_W, double J[][n_W]){
    for (int i = 0; i < n_h; i++){
        unsigned int start_ind = (i*n_prev);
        for (int j = 0; j < n_prev; j++){
            J[i][start_ind + j] = neur[j].h;
        }
    }
}

void evaluate_dh1_dW_LI_L1(unsigned int sample){
    for (int i=0; i<N_NEURONS_L1; i++){
        for (int j=0; j<(N_NEURONS_LI*N_NEURONS_L1); j++){
            if (j >= (i*N_NEURONS_LI) && j < ((i+1) * N_NEURONS_LI)) {
                dh1_dW_LI_L1[i][j] = ((double) training_data[sample][j - (i * N_NEURONS_LI)]) / 255.0 ;
            } else {
                dh1_dW_LI_L1[i][j] = 0.0;
            }
        }
    }
}

void inline evaluate_dh1_dW_LI_L1_sparse(unsigned int sample){
    for (int i=0; i<N_NEURONS_L1; i++){
        unsigned int start_ind = i * N_NEURONS_LI;
        for (int j=0; j<(N_NEURONS_LI); j++){
            dh1_dW_LI_L1[i][start_ind + j] = ((double) training_data[sample][j]) / 255.0 ;
        }
    }
}

void inline evaluate_dh_dh_prev_inc_pre_act(unsigned int n_h, unsigned int n_prev, n_ReLu n_Lh[n_prev], weight_struct_t W[][n_h], double J[][n_prev]){
    for (int i=0; i<n_h; i++){
        for (int j=0; j < n_prev; j++){
            if(n_Lh[j].h > 0.0){
                J[i][j] = W[j][i].w;
            } else {
                J[i][j] = 0.0;
            }
        }
    }
}

void inline sparse_multiply_dL_dh_by_dh_dW_LH_LH_prev(unsigned int rows1, unsigned int cols1, double mat1[][cols1], // dl_dh
                                               unsigned int rows2, unsigned int cols2, double mat2[][cols2], // dh_dW_LH_LH_prev
                                               unsigned int rows_out, unsigned int cols_out, double mat_out[][cols_out]){//dL_dW_LH_LH_prev
    unsigned int n_pre = cols_out/rows2;
    // loop over columns in dh_dW_LH_LH_prev (only one row in dl_dh)
    for (int i = 0; i < rows2; i++){
        unsigned int start_ind = (i * n_pre);
        for (int j = 0; j < n_pre; j++){
            mat_out[0][j + start_ind] =  mat1[0][i] * mat2[i][j + start_ind];
        }
    }
}

void compute_softmax(void){
    //    printf("Computing softmax...\n");
    
    // Compute denominator
    double sft_max_denom = 0.0;
    for (int i=0; i<N_NEURONS_LO; i++){
        sft_max_denom += exp(n_LO[i].z);
    }
    sft_max_denom = 1.0/sft_max_denom;
    
    // Compute softmax
    for (int i=0; i<N_NEURONS_LO; i++){
        n_LO[i].P = exp(n_LO[i].z) * sft_max_denom;
        //        printf("P output %u: %6.12f (logit value: %6.12f) \n", i, n_LO[i].P, n_LO[i].z);
    }
}

// ***********************************************************

void initialise_nn(void){
    unsigned int seed = (unsigned int) time(NULL);
    printf("PRNG initialised using seed: %u\n", seed);
    srand(seed); // initialise RNG so rand() can be used later
    initialise_weight_matrices();
    initialise_gradients_and_jacobians();
    initialise_neurons();
}

void evaluate_forward_pass(uint8_t** dataset, int n){
    // Forward pass
    // For each layer of the network, j is used to index neurons in current layer, and i to index neurons in preceding layer
    
    // Calculate L1 Activations using input
    for (int j=0; j < N_NEURONS_L1; j++){
        n_L1[j].z = 0.0; // reset previous activation to zero
        for (int i=0; i< N_NEURONS_LI; i++){
            n_L1[j].z += w_LI_L1[i][j].w * (( (double) dataset[n][i]) / 255.0);
        }
        // Apply nonlinearity (ReLU)
        n_L1[j].h = MAX(0, n_L1[j].z);
    }
    
    // Calculate L2 Activations
    for (int j=0; j<N_NEURONS_L2; j++){
        n_L2[j].z = 0.0;
        for (int i=0; i< N_NEURONS_L1; i++){
            n_L2[j].z += w_L1_L2[i][j].w * n_L1[i].h;
        }
        // Apply nonlinearity (ReLU)
        n_L2[j].h = MAX(0, n_L2[j].z);
    }
    
    // Calculate L3 Activations
    for (int j=0; j<N_NEURONS_L3; j++){
        n_L3[j].z = 0.0;
        for (int i=0; i< N_NEURONS_L2; i++){
            n_L3[j].z += w_L2_L3[i][j].w * n_L2[i].h;
        }
        // Apply nonlinearity (ReLU)
        n_L3[j].h = MAX(0, n_L3[j].z);
    }
    
    // Calculate L0 Activations (must calculate all weighted-sum inputs before evaluating softmax)
    for (int j=0; j<N_NEURONS_LO; j++){
        n_LO[j].z = 0.0;
        for (int i=0; i<N_NEURONS_L3; i++){
            n_LO[j].z += w_L3_LO[i][j].w * n_L3[i].h;
        }
    }
    
    // Compute softmax of output layer
    compute_softmax();
}

double compute_xent_loss(uint8_t label){
    //    printf("Computing Loss...\n");
    double L = 0.0;
    L = -1 * log(n_LO[label].P);
    
    //    printf("Loss: %.30f\n", L);
    return L;
}

void evaluate_backward_pass(uint8_t label, unsigned int input_class_index){
        
    // Update all Gradients/Jacobians
    evaluate_dL_dP(label);
    evaluate_dP_dhO();
    
    // Evaluate derivatives of layerwise activations wrt weight matrix connecting previous layer
    evaluate_dh_dW_Lh_Lprev(N_NEURONS_LO, N_NEURONS_L3, n_L3, N_NEURONS_LO*N_NEURONS_L3, dhO_dW_L3_LO); // dhO_dW_L3_LO
    evaluate_dh_dW_Lh_Lprev(N_NEURONS_L3, N_NEURONS_L2, n_L2, N_NEURONS_L3*N_NEURONS_L2, dh3_dW_L2_L3); // dh3_dW_L2_L3
    evaluate_dh_dW_Lh_Lprev(N_NEURONS_L2, N_NEURONS_L1, n_L1, N_NEURONS_L2*N_NEURONS_L1, dh2_dW_L1_L2); // dh2_dW_L1_L2
    evaluate_dh1_dW_LI_L1(input_class_index); // dh1_dW_LI_L1
    
    // Evaluate derivatives of layerwise activations wrt activation of previous layer
    evaluate_dh_dh_prev_inc_pre_act(N_NEURONS_LO, N_NEURONS_L3, n_L3, w_L3_LO, dhO_dh3); // dhO_dh3
    evaluate_dh_dh_prev_inc_pre_act(N_NEURONS_L3, N_NEURONS_L2, n_L2, w_L2_L3, dh3_dh2); // dh3_dh2
    evaluate_dh_dh_prev_inc_pre_act(N_NEURONS_L2, N_NEURONS_L1, n_L1, w_L1_L2, dh2_dh1); // dh2_dh1
    
    // Evaluate Jacobians wrt L
    // dL_dW_L3_LO = (dL_dP * dP_dhO) * dhO_dW_L3_LO
    double_matrix_multiply(1, N_NEURONS_LO, dL_dP,
                           N_NEURONS_LO, N_NEURONS_LO, dP_dhO,
                           1, N_NEURONS_LO, dL_dhO);
    double_matrix_multiply(1, N_NEURONS_LO, dL_dhO,
                           N_NEURONS_LO, N_NEURONS_L3*N_NEURONS_LO, dhO_dW_L3_LO,
                           1, N_NEURONS_L3*N_NEURONS_LO, dL_dW_L3_LO);
    
    // dL_dW_L2_L3 = (dL_dhO * dhO_dh3) * dh3_dW_L2_L3
    double_matrix_multiply(1, N_NEURONS_LO, dL_dhO,
                           N_NEURONS_LO, N_NEURONS_L3, dhO_dh3,
                           1, N_NEURONS_L3, dL_dh3);
    double_matrix_multiply(1, N_NEURONS_L3, dL_dh3,
                           N_NEURONS_L3, N_NEURONS_L2*N_NEURONS_L3, dh3_dW_L2_L3,
                           1, N_NEURONS_L2*N_NEURONS_L3, dL_dW_L2_L3);
    
    // dL_dW_L1_L2 = (dL_h3 * dh3_dh2) * dh2_dW_L1_L2
    double_matrix_multiply(1, N_NEURONS_L3, dL_dh3,
                           N_NEURONS_L3, N_NEURONS_L2, dh3_dh2,
                           1, N_NEURONS_L2, dL_dh2);
    double_matrix_multiply(1, N_NEURONS_L2, dL_dh2,
                           N_NEURONS_L2, N_NEURONS_L1*N_NEURONS_L2, dh2_dW_L1_L2,
                           1, N_NEURONS_L1*N_NEURONS_L2, dL_dW_L1_L2);
    
    // dL_dW_LI_L1 = (dL_dh2 * dh2_dh1) * dh1_dW_LI_L1
    double_matrix_multiply(1, N_NEURONS_L2, dL_dh2,
                           N_NEURONS_L2, N_NEURONS_L1, dh2_dh1,
                           1, N_NEURONS_L1, dL_dh1);
    double_matrix_multiply(1, N_NEURONS_L1, dL_dh1,
                           N_NEURONS_L1, N_NEURONS_LI*N_NEURONS_L1, dh1_dW_LI_L1,
                           1, N_NEURONS_LI*N_NEURONS_L1, dL_dW_LI_L1);
    
}

void evaluate_backward_pass_sparse(uint8_t label, unsigned int input_class_index){
    // Update all Gradients/Jacobians via error backpropagation
    
    // Evaluate gradient of loss function wrt to output layer activity
    evaluate_dL_dP(label);
    evaluate_dP_dhO();
    
    // Evaluate derivatives of neuron activation wrt to weights
    evaluate_dh_dW_Lh_Lprev_sparse(N_NEURONS_LO, N_NEURONS_L3, n_L3, N_NEURONS_LO*N_NEURONS_L3, dhO_dW_L3_LO); // dhO_dW_L3_LO
    evaluate_dh_dW_Lh_Lprev_sparse(N_NEURONS_L3, N_NEURONS_L2, n_L2, N_NEURONS_L3*N_NEURONS_L2, dh3_dW_L2_L3); // dh3_dW_L2_L3
    evaluate_dh_dW_Lh_Lprev_sparse(N_NEURONS_L2, N_NEURONS_L1, n_L1, N_NEURONS_L2*N_NEURONS_L1, dh2_dW_L1_L2); // dh2_dW_L1_L2
    evaluate_dh1_dW_LI_L1_sparse(input_class_index); // dh1_dW_LI_L1
    
    // Evaluate derivatives of layerwise activations wrt activation of previous layer
    evaluate_dh_dh_prev_inc_pre_act(N_NEURONS_LO, N_NEURONS_L3, n_L3, w_L3_LO, dhO_dh3); // dhO_dh3
    evaluate_dh_dh_prev_inc_pre_act(N_NEURONS_L3, N_NEURONS_L2, n_L2, w_L2_L3, dh3_dh2); // dh3_dh2
    evaluate_dh_dh_prev_inc_pre_act(N_NEURONS_L2, N_NEURONS_L1, n_L1, w_L1_L2, dh2_dh1); // dh2_dh1
    
    // Evaluate gradients wrt L
    // dL_dW_L3_LO = (dL_dP * dP_dhO) * dhO_dW3O
    double_matrix_multiply(1, N_NEURONS_LO, dL_dP,
                           N_NEURONS_LO, N_NEURONS_LO, dP_dhO,
                           1, N_NEURONS_LO, dL_dhO);
    sparse_multiply_dL_dh_by_dh_dW_LH_LH_prev(1, N_NEURONS_LO, dL_dhO,
                                              N_NEURONS_LO, N_NEURONS_L3*N_NEURONS_LO, dhO_dW_L3_LO,
                                              1, N_NEURONS_L3*N_NEURONS_LO, dL_dW_L3_LO);
    
    // dL_dW_L2_L3 = (dL_dhO * dhO_dh3) * dh3_dW_L2_L3
    double_matrix_multiply(1, N_NEURONS_LO, dL_dhO,
                           N_NEURONS_LO, N_NEURONS_L3, dhO_dh3,
                           1, N_NEURONS_L3, dL_dh3);
    sparse_multiply_dL_dh_by_dh_dW_LH_LH_prev(1, N_NEURONS_L3, dL_dh3,
                                              N_NEURONS_L3, N_NEURONS_L2*N_NEURONS_L3, dh3_dW_L2_L3,
                                              1, N_NEURONS_L2*N_NEURONS_L3, dL_dW_L2_L3);
    
    // dL_dW_L1_L2 = (dL_h3 * dh3_dh2) * dh2_dW_L1_L2
    double_matrix_multiply(1, N_NEURONS_L3, dL_dh3,
                           N_NEURONS_L3, N_NEURONS_L2, dh3_dh2,
                           1, N_NEURONS_L2, dL_dh2);
    sparse_multiply_dL_dh_by_dh_dW_LH_LH_prev(1, N_NEURONS_L2, dL_dh2,
                                              N_NEURONS_L2, N_NEURONS_L1*N_NEURONS_L2, dh2_dW_L1_L2,
                                              1, N_NEURONS_L1*N_NEURONS_L2, dL_dW_L1_L2);
    
    // dL_dW_LI_L1 = (dL_dh2 * dh2_dh1) * dh1_dW_LI_L1
    double_matrix_multiply(1, N_NEURONS_L2, dL_dh2,
                           N_NEURONS_L2, N_NEURONS_L1, dh2_dh1,
                           1, N_NEURONS_L1, dL_dh1);
    sparse_multiply_dL_dh_by_dh_dW_LH_LH_prev(1, N_NEURONS_L1, dL_dh1,
                                              N_NEURONS_L1, N_NEURONS_LI*N_NEURONS_L1, dh1_dW_LI_L1,
                                              1, N_NEURONS_LI*N_NEURONS_L1, dL_dW_LI_L1);
}

void evaluate_weight_updates(void){
    // dW_L3_LO
    for (int i = 0; i < N_NEURONS_L3; i++){
        for (int j=0; j < N_NEURONS_LO; j++){
            w_L3_LO[i][j].dw += dL_dW_L3_LO[0][(i + (N_NEURONS_L3 * j))];
        }
    }

    // dW_L2_L3
    for (int i = 0; i < N_NEURONS_L2 ; i++){
        for (int j=0; j < N_NEURONS_L3; j++){
            w_L2_L3[i][j].dw +=  dL_dW_L2_L3[0][(i + (N_NEURONS_L2 * j))];
        }
    }
    
    // dW_L1_L2
    for (int i = 0; i < N_NEURONS_L1 ; i++){
        for (int j=0; j < N_NEURONS_L2; j++){
            w_L1_L2[i][j].dw +=  dL_dW_L1_L2[0][(i + (N_NEURONS_L1 * j))];
        }
    }
    
    // dW_LI_L1
    for (int i = 0; i < N_NEURONS_LI ; i++){
        for (int j=0; j < N_NEURONS_L1; j++){
            w_LI_L1[i][j].dw += dL_dW_LI_L1[0][(i + (N_NEURONS_LI * j))];
        }
    }
}

double evaluate_testing_accuracy(void){
    double prediction_accuracy = 0.0;
    unsigned int correct_predictions = 0;
    
    for (int i = 0; i< N_TESTING_SET; i++){
        // Compute forward pass on sample from test set
        evaluate_forward_pass(testing_data, i);
        // Find prediction from softmax layer of network
        unsigned int max_ind = 0;
        for (int j=1; j<N_NEURONS_LO; j++){
            if (n_LO[j].P > n_LO[max_ind].P){
                max_ind = j;
            }
        }
        // max_ind now holds predicted class from network
        if (max_ind == testing_labels[i]){
            correct_predictions++;
        }
    }
    
    // Compute accuracy
    prediction_accuracy = (double) correct_predictions / N_TESTING_SET;
    
    return prediction_accuracy;
}

