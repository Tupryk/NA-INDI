// GENERATED FILE FROM MODEL ./models/mlp_zeroed.pth
// Info: Null neural network
#include "nn_null.h"
#include "nn_utils.h"


struct Dataset dataset = {
    .current_size = 0,
    .oldest_entry = 0,
};

struct OnlineNeuralNetwork mlp_zeroed = {
    .hidden_weight = { { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. } },
    .hidden_bias = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
    .hidden_pre_activation = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
    .hidden_activation = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
    .output_weight = { { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. }, { 0., 0., 0., 0., 0., 0. } },
    .output_bias = { 0., 0., 0., 0., 0., 0. },
    .output_pre_activation = { 0., 0., 0., 0., 0., 0. },
    .output_activation = { 0., 0., 0., 0., 0., 0. },
};

void add_dataset_entry(float input[INPUT_SIZE], float output[OUTPUT_SIZE])
{
    int change_idx = 0;
    if (dataset.current_size < DATASET_MAX_SIZE)
    {
        change_idx = dataset.current_size;
        dataset.current_size++;
    }
    else
    {
        change_idx = dataset.oldest_entry;
        dataset.oldest_entry++;
        if (dataset.oldest_entry >= DATASET_MAX_SIZE)
        {
            dataset.oldest_entry = 0;
        }
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        dataset.inputs[change_idx][i] = input[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        dataset.outputs[change_idx][i] = output[i];
    }
}

const float* nn_forward(float input[INPUT_SIZE]) {
    // Forward
    layer(INPUT_SIZE, HIDDEN_SIZE, input, mlp_zeroed.hidden_weight, mlp_zeroed.hidden_bias,
        mlp_zeroed.hidden_pre_activation, mlp_zeroed.hidden_activation, 1);
    layer(HIDDEN_SIZE, OUTPUT_SIZE, mlp_zeroed.hidden_activation, mlp_zeroed.output_weight, mlp_zeroed.output_bias,
        mlp_zeroed.output_pre_activation, mlp_zeroed.output_activation, 0);
    return mlp_zeroed.output_activation;
};

float ReLU_derivative(float x) {
    return (x > 0) ? 1.0 : 0.0;
}

void backprop(float lr)
{
    // Init gradient descent step
    float hidden_weight_update[INPUT_SIZE][HIDDEN_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            hidden_weight_update[i][j] = 0.f;
    }
    float hidden_bias_update[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_bias_update[i] = 0.f;
    }
    float output_weight_update[HIDDEN_SIZE][OUTPUT_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output_weight_update[i][j] = 0.f;
        }
    }
    float output_bias_update[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_bias_update[i] = 0.f;
    }

    for (int d = 0; d < dataset.current_size; d++)
    {
        nn_forward(dataset.inputs[d]);
    
        // Compute errors on output layer
        float output_layer_error[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            output_layer_error[i] = mlp_zeroed.output_activation[i] - dataset.outputs[d][i];
        }
    
        // Compute errors of hidden layer
        float hidden_layer_error[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            hidden_layer_error[i] = 0.f;
            for (int e = 0; e < OUTPUT_SIZE; e++)
            {
                hidden_layer_error[i] += mlp_zeroed.output_weight[i][e] * output_layer_error[e];
            }
            hidden_layer_error[i] *= ReLU_derivative(mlp_zeroed.hidden_pre_activation[i]);
        }
    
        // Update weights
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN_SIZE; j++)
            {
                float weight_update = (
                    output_layer_error[i] *
                    mlp_zeroed.hidden_activation[j]
                );
                output_weight_update[j][i] += weight_update;
            }
        }
    
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                float weight_update = (
                    hidden_layer_error[i] *
                    dataset.inputs[0][j] *
                    ReLU_derivative(mlp_zeroed.hidden_activation[i])
                );
                hidden_weight_update[j][i] += weight_update;
            }
        }
    
        // Update biases
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            float bias_update = (
                output_layer_error[i]
            );
            output_bias_update[i] += bias_update;
        }
    
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            float bias_update = (
                hidden_layer_error[i] *
                ReLU_derivative(mlp_zeroed.hidden_activation[i])
            );
            hidden_bias_update[i] += bias_update;
        }
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            mlp_zeroed.hidden_weight[i][j] -= hidden_weight_update[i][j] * lr;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        mlp_zeroed.hidden_bias[i] -= hidden_bias_update[i] * lr;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            mlp_zeroed.output_weight[i][j] -= output_weight_update[i][j] * lr;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        mlp_zeroed.output_bias[i] -= output_bias_update[i] * lr;
    }
}
