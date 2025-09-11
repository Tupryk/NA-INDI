// GENERATED FILE FROM MODEL ./models/mlp_zeroed.pth
// Info: Null neural network
#ifndef __GEN_NN__
#define __GEN_NN__

#define DATASET_MAX_SIZE 1

#define INPUT_SIZE 19
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 6

struct Dataset
{
    int current_size;
    int oldest_entry;
    float inputs[DATASET_MAX_SIZE][INPUT_SIZE];
    float outputs[DATASET_MAX_SIZE][OUTPUT_SIZE];
};

struct OnlineNeuralNetwork
{
    float hidden_weight[INPUT_SIZE][HIDDEN_SIZE];
    float hidden_bias[HIDDEN_SIZE];
    float hidden_pre_activation[HIDDEN_SIZE];
    float hidden_activation[HIDDEN_SIZE];
    float output_weight[HIDDEN_SIZE][OUTPUT_SIZE];
    float output_bias[OUTPUT_SIZE];
    float output_pre_activation[OUTPUT_SIZE];
    float output_activation[OUTPUT_SIZE];
};

void add_dataset_entry(float input[INPUT_SIZE], float output[OUTPUT_SIZE]);
const float* nn_forward(float input[INPUT_SIZE]);
void backprop(float lr);

#endif
