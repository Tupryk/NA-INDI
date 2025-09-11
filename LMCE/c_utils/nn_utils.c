#include "nn_utils.h"


static float ReLU(float num) {
	return (num > 0) ? num : 0.f;
}

static float LeakyReLU(float num) {
	return (num > 0) ? num : num * 0.01f;
}

void layer(int rows, int cols, float in[rows], float layer_weight[rows][cols], float layer_bias[cols],
		   float pre_output[cols], float output[cols], int use_activation) {
	for(int ii = 0; ii < cols; ii++) {
		pre_output[ii] = 0;
		for (int jj = 0; jj < rows; jj++) {
			pre_output[ii] += in[jj] * layer_weight[jj][ii];
		}
		pre_output[ii] += layer_bias[ii];
		if (use_activation == 1) {
			output[ii] = ReLU(pre_output[ii]);
		} else if (use_activation == 2) {
			output[ii] = LeakyReLU(pre_output[ii]);
		} else {
			output[ii] = pre_output[ii];
		}
	}
}
