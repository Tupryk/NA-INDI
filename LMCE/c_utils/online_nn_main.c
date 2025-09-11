#include <stdio.h>

#include "nn_null.h"

int main()
{
	float input[INPUT_SIZE];
	float output[OUTPUT_SIZE];

	for (int i = 0; i < INPUT_SIZE; i++) {
		input[i] = i;
	}
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		output[i] = i;
		printf("%f, ", output[i]);
	}
	printf("\n");

	add_dataset_entry(input, output);
	for (int i = 0; i < 100000; i++) {
		backprop(0.0003f);
	}

	const float *model_output = nn_forward(input);
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		printf("%f, ", model_output[i]);
	}
	printf("\n");
	return 0;
}
