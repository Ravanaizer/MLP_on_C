#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

static int same_shape(const Tensor *a, const Tensor *b) {
  if (a->ndim != b->ndim)
    return 0;
  for (int i = 0; i < a->ndim; i++)
    if (a->shape[i] != b->shape[i])
      return 0;
  return 1;
}

Tensor *tensor_new(int ndim, const int *shape) {
  if (ndim < 1 || ndim > MAX_DIMS)
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  t->ndim = ndim;
  t->size = 1;
  for (int i = 0; i < ndim; i++) {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }

  t->data = malloc(t->size * sizeof(float));
  if (!t->data) {
    free(t);
    return NULL;
  }
  return t;
}

void tensor_free(Tensor *t) {
  if (!t)
    return;
  free(t->data);
  free(t);
}

void tensor_add(Tensor *first, const Tensor *second) {
  assert(same_shape(first, second));
  for (int i = 0; i < first->size; i++)
    first->data[i] += second->data[i];
}

void tensor_sub(Tensor *first, const Tensor *second) {
  assert(same_shape(first, second));
  for (int i = 0; i < first->size; i++)
    first->data[i] -= second->data[i];
}

void tensor_mul(Tensor *first, const Tensor *second) {
  assert(same_shape(first, second));
  for (int i = 0; i < first->size; i++)
    first->data[i] *= second->data[i];
}

void tensor_div(Tensor *first, const Tensor *second) {
  assert(same_shape(first, second));
  for (int i = 0; i < first->size; i++)
    first->data[i] /= second->data[i];
}

float tensor_sum(const Tensor *tensor) {
  float sum = 0;
  for (int i = 0; i < tensor->size; i++)
    sum += tensor->data[i];
  return sum;
}

void tensor_zero(Tensor *tensor) {
  for (int i = 0; i < tensor->size; i++)
    tensor->data[i] = 0;
}

void tensor_rand_range(Tensor *tensor, float lo, float hi) {
  for (int i = 0; i < tensor->size; i++) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    tensor->data[i] = lo + r * (hi - lo);
  }
}

int tensor_load(Tensor *t, FILE *f) {
  size_t n = (size_t)t->size;
  return fread(t->data, sizeof(float), n, f) == n ? 0 : -1;
}

int tensor_save(const Tensor *t, FILE *f) {
  size_t n = (size_t)t->size;
  return fwrite(t->data, sizeof(float), n, f) == n ? 0 : -1;
}
