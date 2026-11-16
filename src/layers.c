#include <stdlib.h>
#include <string.h>

#include "layers.h"
#include "tensor.h"

void linear_forward(Linear *self, const Tensor *x, Tensor *y) {
  int in = self->in_features;
  int out = self->out_features;

  memcpy(self->x, x->data, in * sizeof(float));

  for (int i = 0; i < out; i++) {
    float s = self->bias->data[i];
    for (int j = 0; j < in; j++) {
      s += self->weight->data[i * in + j] * x->data[j];
    }
    y->data[i] = s;
  }
}

void linear_grad(Linear *self, const Tensor *dy, Tensor *dx) {
  int in = self->in_features;
  int out = self->out_features;

  for (int j = 0; j < in; j++)
    dx->data[j] = 0;

  for (int i = 0; i < out; i++) {
    float g = dy->data[i];
    self->db->data[i] += g;
    for (int j = 0; j < in; j++) {
      self->dw->data[i * in + j] += g * self->x[j];
      dx->data[j] += self->weight->data[i * in + j] * g;
    }
  }
}

Linear *linear_new(int in, int out) {
  Linear *l = calloc(1, sizeof(Linear));
  if (!l) return NULL;

  l->in_features = in;
  l->out_features = out;
  l->weight = tensor_new(2, (int[]){out, in});
  l->bias   = tensor_new(1, (int[]){out});
  l->dw     = tensor_new(2, (int[]){out, in});
  l->db     = tensor_new(1, (int[]){out});
  l->x      = malloc(in * sizeof(float));

  if (!l->weight || !l->bias || !l->dw || !l->db || !l->x) {
    linear_free(l);
    return NULL;
  }

  tensor_zero(l->dw);
  tensor_zero(l->db);
  return l;
}

void linear_free(Linear *l) {
  if (!l) return;
  if (l->weight) tensor_free(l->weight);
  if (l->bias)   tensor_free(l->bias);
  if (l->dw)     tensor_free(l->dw);
  if (l->db)     tensor_free(l->db);
  free(l->x);
  free(l);
}

void ReLU(Tensor *tensor) {
  int size = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    size *= tensor->shape[i];
  }

  for (int i = 0; i < size; i++) {
    tensor->data[i] = tensor->data[i] > 0 ? tensor->data[i] : 0;
  }
}
