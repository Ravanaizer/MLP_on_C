#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>

#define MAX_DIMS 4

typedef struct {
  float *data;
  int shape[MAX_DIMS];
  int ndim;
  int size;
} Tensor;

Tensor *tensor_new(int ndim, const int *shape);
void tensor_free(Tensor *tensor);
void tensor_zero(Tensor *tensor);
void tensor_rand_range(Tensor *tensor, float lo, float hi);

void tensor_add(Tensor *first, const Tensor *second);
void tensor_sub(Tensor *first, const Tensor *second);
void tensor_mul(Tensor *first, const Tensor *second);
void tensor_div(Tensor *first, const Tensor *second);
float tensor_sum(const Tensor *tensor);

int tensor_load(Tensor *t, FILE *f);
int tensor_save(const Tensor *t, FILE *f);

#endif
