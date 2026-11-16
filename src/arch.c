#include "arch.h"
#include "layers.h"
#include "tensor.h"

Arch *arch_new(int size, int *neurons) {
  Arch *new_arch = malloc(sizeof(Arch));
  new_arch->size = size;

  a->neurons = malloc(size * sizeof(int));
  a->layers = calloc(size - 1, sizeof(Linear *));
  a->act = calloc(size, sizeof(Tensor *));

  if (!a->neurons || !a->layers || !a->act)
    goto fail;

  for (int i = 0; i < size; i++)
    a->neurons[i] = neurons[i];

  for (int i = 0; i < size - 1; i++) {
    a->layers[i] = linear_new(neurons[i], neurons[i + 1]);
    if (!a->layers[i])
      goto fail;
  }

  for (int i = 0; i < size; i++) {
    a->act[i] = tensor_new(1, (int[]){neurons[i]});
    if (!a->act[i])
      goto fail;
  }
  return a;

fail:
  arch_free(a);
  return NULL;
}

void arch_free(Arch *arch) {
  for (int i = 0; i < arch->size; i++) {
    tensor_free(arch->act[i]);
  }
  for (int i = 0; i < arch->size - 1; i++) {
    layers_free(arch->layers[i]);
  }
  free(arch->neurons);
  free(arch->act);
  free(arch->layers);
  free(arch);
}

Tensor *arch_forward(Arch *a) {
  for (int i = 0; i < a->size - 1; i++) {
    linear_forward(a->layers[i], a->act[i], a->act[i + 1]);
    if (i < a->size - 2)
      ReLU(a->act[i + 1]);
  }
  return a->act[a->size - 1];
}

float loss_MSE(const float *y, const float *target, int n) {
  float loss = 0;
  for (int i = 0; i < n; i++) {
    loss += (target[i] - y[i]) * (target[i] - y[i]);
  }
  loss /= n;
  return loss;
}

void loss_MSE_grad(const float *y, const float *target, float *dy, int n) {
  for (int i = 0; i < n; i++)
    dy[i] = 2.0f * (y[i] - target[i]) / n;
}
