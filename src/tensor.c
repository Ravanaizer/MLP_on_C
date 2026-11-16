#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

Tensor *tensor_new(int ndim, const int *shape) {
  if (ndim < 1 || ndim > MAX_DIMS) return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t) return NULL;

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
  if (!t) return;
  free(t->data);
  free(t);
}

void tensor_add(Tensor *first, Tensor *second) {
  assert(same_shape(first, second));

  int size = 1;
  for (int i = 0; i < first->ndim; i++) {
    size *= first->shape[i];
  }

  for (int i = 0; i < size; i++) {
    first->data[i] += second->data[i];
  }
}

void tensor_sub(Tensor *first, Tensor *second) {
  assert(same_shape(first, second));

  int size = 1;
  for (int i = 0; i < first->ndim; i++) {
    size *= first->shape[i];
  }

  for (int i = 0; i < size; i++) {
    first->data[i] -= second->data[i];
  }
}

void tensor_div(Tensor *first, Tensor *second) {
  assert(same_shape(first, second));

  int size = 1;
  for (int i = 0; i < first->ndim; i++) {
    size *= first->shape[i];
  }

  for (int i = 0; i < size; i++) {
    first->data[i] *= second->data[i];
  }
}

void tensor_mul(Tensor *first, Tensor *second) {
  assert(same_shape(first, second));

  int size = 1;
  for (int i = 0; i < first->ndim; i++) {
    size *= first->shape[i];
  }

  for (int i = 0; i < size; i++) {
    first->data[i] /= second->data[i];
  }
}

float tensor_sum(Tensor *tensor) {
  int size = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    size *= tensor->shape[i];
  }

  float sum = 0;

  for (int i = 0; i < size; i++) {
    sum += tensor->data[i];
  }
}

void tensor_zero(Tensor *tensor) {
  int size = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    size *= tensor->shape[i];
  }

  for (int i = 0; i < size; i++) {
    tensor->data[i] = 0;
  }
}

void tensor_rand_range(Tensor *tensor, float lo, float hi) {
  int size = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    size *= tensor->shape[i];
  }

  for (int i = 0; i < size; i++) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    tensor->data[i] = lo + r * (hi - lo);
  }
}

static int same_shape(const Tensor *a, const Tensor *b) {
  if (a->ndim != b->ndim)
    return 0;
  for (int i = 0; i < a->ndim; i++)
    if (a->shape[i] != b->shape[i])
      return 0;
  return 1;
}

static int load_tensor_from_file(Tensor *t, FILE *f) {
  size_t n = numel(t);
  return fread(t->data, sizeof(float), n, f) == n ? 0 : -1;
}
