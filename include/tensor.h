#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMS 4

typedef struct {
  float *data;
  int shape[MAX_DIMS];
  int ndim;
} Tensor;


void tensor_add(Tensor *first, Tensor *second);
void tensor_sub(Tensor *first, Tensor *second);
void tensor_mul(Tensor *first, Tensor *second);
void tensor_div(Tensor *first, Tensor *second);

float tensor_sum(Tensor *tensor);

Tensor *tensor_new(int ndim, const int *shape);
void tensor_free(Tensor *tensor);
void tensor_zero(Tensor *tensor);
void tensor_rand(Tensor *tensor, float lo, float hi);

static int same_shape(const Tensor *a, const Tensor *b);
static int load_tensor_from_file(Tensor *t, FILE *f);

#endif
