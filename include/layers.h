#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

typedef struct {
  Tensor *weight;
  Tensor *bias;
  Tensor *dw;
  Tensor *db;
  float *x;
  int in_features;
  int out_features;
} Linear;

void linear_forward(Linear *self, const Tensor *x, Tensor *y);
void linear_grad(Linear *self, const Tensor *dy, Tensor *dx);
Linear *linear_new(int in, int out);
void linear_free(Linear *layer);

void ReLU(Tensor *tensor);


#endif
