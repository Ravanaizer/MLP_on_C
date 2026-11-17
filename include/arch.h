#ifndef ARCH_H
#define ARCH_H

#include "layers.h"
#include "tensor.h"

typedef struct {
  int size;         // длина neurons (слоёв Linear на 1 меньше)
  int *neurons;
  Linear **layers;  // size - 1
  Tensor **act;     // size: act[0] вход, act[size-1] выход
  Tensor **grad;    // size: dL/d(act[i])
} Arch;

Arch *arch_new(int size, const int *neurons);
void arch_free(Arch *a);
void arch_init_random(Arch *a);
Tensor *arch_forward(Arch *a);
// Перед вызовом grad[size-1] должен содержать dL/dy (см. loss_MSE_grad)
void arch_backward(Arch *a);
void arch_step(Arch *a, float alpha, int batch);

float loss_MSE(const float *y, const float *target, int n);
void loss_MSE_grad(const float *y, const float *target, float *dy, int n);

void linear_step(Linear *l, float alpha, int batch);

#endif
