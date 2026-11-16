#ifndef ARCH_H
#define ARCH_H

#include "layers.h"
#include "tensor.h"

typedef struct {
  int size;
  int *neurons;
  Linear **layers;
  Tensor **act;
} Arch;

Arch *arch_new(int size, int *neurons);
void arch_free(Arch *arch);
Tensor *arch_forward(Arch *a);

float loss_MSE(float *avg_y, int size, float y);
void loss_MSE_grad(const float *y, const float *target, float *dy, int n);



#endif
