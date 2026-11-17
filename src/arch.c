#include <math.h>
#include <stdlib.h>

#include "arch.h"
#include "layers.h"
#include "tensor.h"

Arch *arch_new(int size, const int *neurons) {
  if (size < 2)
    return NULL;

  Arch *a = calloc(1, sizeof(Arch));
  if (!a)
    return NULL;
  a->size = size;

  a->neurons = malloc(size * sizeof(int));
  a->layers = calloc(size - 1, sizeof(Linear *));
  a->act = calloc(size, sizeof(Tensor *));
  a->grad = calloc(size, sizeof(Tensor *));
  if (!a->neurons || !a->layers || !a->act || !a->grad)
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
    a->grad[i] = tensor_new(1, (int[]){neurons[i]});
    if (!a->act[i] || !a->grad[i])
      goto fail;
  }
  return a;

fail:
  arch_free(a);
  return NULL;
}

void arch_free(Arch *a) {
  if (!a)
    return;
  for (int i = 0; i < a->size; i++) {
    if (a->act)
      tensor_free(a->act[i]);
    if (a->grad)
      tensor_free(a->grad[i]);
  }
  if (a->layers)
    for (int i = 0; i < a->size - 1; i++)
      linear_free(a->layers[i]);
  free(a->neurons);
  free(a->act);
  free(a->grad);
  free(a->layers);
  free(a);
}

void arch_init_random(Arch *a) {
  for (int i = 0; i < a->size - 1; i++) {
    float bound = 1.0f / sqrtf((float)a->layers[i]->in_features);
    tensor_rand_range(a->layers[i]->weight, -bound, bound);
    tensor_zero(a->layers[i]->bias);
  }
}

Tensor *arch_forward(Arch *a) {
  for (int i = 0; i < a->size - 1; i++) {
    linear_forward(a->layers[i], a->act[i], a->act[i + 1]);
    if (i < a->size - 2)
      ReLU(a->act[i + 1]);
  }
  return a->act[a->size - 1];
}

void arch_backward(Arch *a) {
  for (int i = a->size - 2; i >= 0; i--) {
    linear_grad(a->layers[i], a->grad[i + 1], a->grad[i]);
    if (i > 0) {
      for (int k = 0; k < a->neurons[i]; k++)
        if (a->act[i]->data[k] <= 0)
          a->grad[i]->data[k] = 0;
    }
  }
}

void arch_step(Arch *a, float alpha, int batch) {
  for (int i = 0; i < a->size - 1; i++)
    linear_step(a->layers[i], alpha, batch);
}

float loss_MSE(const float *y, const float *target, int n) {
  float loss = 0;
  for (int i = 0; i < n; i++)
    loss += (target[i] - y[i]) * (target[i] - y[i]);
  return loss / n;
}

void loss_MSE_grad(const float *y, const float *target, float *dy, int n) {
  for (int i = 0; i < n; i++)
    dy[i] = 2.0f * (y[i] - target[i]) / n;
}

void linear_step(Linear *l, float alpha, int batch) {
  int n_w = l->out_features * l->in_features;
  for (int i = 0; i < n_w; i++) {
    l->weight->data[i] -= alpha * l->dw->data[i] / batch;
    l->dw->data[i] = 0;
  }
  for (int i = 0; i < l->out_features; i++) {
    l->bias->data[i] -= alpha * l->db->data[i] / batch;
    l->db->data[i] = 0;
  }
}
