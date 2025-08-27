#include "nn_utils.h"


static float ReLU(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

static float LeakyReLU(float num) {
	if (num > 0) {
		return num;
	} else {
		return num * 0.01f;
	}
}

void layer(int rows, int cols, float in[rows], float layer_weight[rows][cols], float layer_bias[cols],
			float output[cols], int use_activation) {
	for(int ii = 0; ii < cols; ii++) {
		output[ii] = 0;
		for (int jj = 0; jj < rows; jj++) {
			output[ii] += in[jj] * layer_weight[jj][ii];
		}
		output[ii] += layer_bias[ii];
		if (use_activation == 1) {
			output[ii] = ReLU(output[ii]);
		} else if (use_activation == 2) {
			output[ii] = LeakyReLU(output[ii]);
		}
		
	}
}
