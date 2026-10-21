#ifndef __NN_H__
#define __NN_H__

void layer(int rows, int cols, float in[rows], float layer_weight[rows][cols], float layer_bias[cols], float pre_output[cols], float output[cols], int use_activation);
void sum_vectors(int vec_len, float vec_a[vec_len], float vec_b[vec_len], float vec_out[vec_len]);

#endif