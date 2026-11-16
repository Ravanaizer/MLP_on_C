#include <stdio.h>
#include <stdlib.h>

#include "layers.h"
#include "tensor.h"

#define N_LAYERS 4

// 784 -> 128 -> 64 -> 32 -> 10
static const int SIZES[N_LAYERS + 1] = {784, 128, 64, 32, 10};

// Число элементов тензора (произведение shape), не зависит от поля size
static size_t numel(const Tensor *t) {
  size_t n = 1;
  for (int i = 0; i < t->ndim; i++)
    n *= (size_t)t->shape[i];
  return n;
}

// Читает numel(t) float'ов из файла прямо в t->data
static int load_tensor(Tensor *t, FILE *f) {
  size_t n = numel(t);
  return fread(t->data, sizeof(float), n, f) == n ? 0 : -1;
}

int main(int argc, char **argv) {
  const char *weights_path = argc > 1 ? argv[1] : "../mnist_weights.bin";
  const char *input_path = argc > 2 ? argv[2] : "../input.bin";

  Linear *layers[N_LAYERS] = {NULL};
  Tensor *act[N_LAYERS + 1] = {NULL}; // act[0] - вход, act[N_LAYERS] - выход
  FILE *f = NULL;
  int status = 1;

  // --- создаём слои и буферы активаций ---
  for (int i = 0; i < N_LAYERS; i++) {
    layers[i] = linear_new(SIZES[i], SIZES[i + 1]);
    if (!layers[i]) {
      fprintf(stderr, "Ошибка: не удалось создать слой %d\n", i + 1);
      goto cleanup;
    }
  }
  for (int i = 0; i <= N_LAYERS; i++) {
    act[i] = tensor_new(1, (int[]){SIZES[i]});
    if (!act[i]) {
      fprintf(stderr, "Ошибка: не удалось создать буфер %d\n", i);
      goto cleanup;
    }
  }

  // --- загружаем веса: weight, bias для каждого слоя подряд ---
  f = fopen(weights_path, "rb");
  if (!f) {
    perror(weights_path);
    goto cleanup;
  }
  for (int i = 0; i < N_LAYERS; i++) {
    if (load_tensor(layers[i]->weight, f) != 0 ||
        load_tensor(layers[i]->bias, f) != 0) {
      fprintf(stderr, "Ошибка: файл весов слишком короткий (слой %d)\n", i + 1);
      goto cleanup;
    }
  }
  // если в файле остались данные, архитектура не совпадает с экспортом
  if (fgetc(f) != EOF) {
    fprintf(stderr, "Ошибка: в файле весов больше данных, чем ждёт сеть\n");
    goto cleanup;
  }
  fclose(f);
  f = NULL;

  // --- загружаем входную картинку (784 float32) ---
  f = fopen(input_path, "rb");
  if (!f) {
    perror(input_path);
    goto cleanup;
  }
  if (load_tensor(act[0], f) != 0) {
    fprintf(stderr, "Ошибка: во входном файле меньше %d чисел\n", SIZES[0]);
    goto cleanup;
  }
  fclose(f);
  f = NULL;

  // --- forward: linear -> ReLU на скрытых слоях, на последнем только linear ---
  for (int i = 0; i < N_LAYERS; i++) {
    linear_forward(layers[i], act[i], act[i + 1]);
    if (i < N_LAYERS - 1)
      ReLU(act[i + 1]);
  }

  // --- вывод логитов и предсказание ---
  Tensor *out = act[N_LAYERS];
  int best = 0;
  for (int i = 0; i < SIZES[N_LAYERS]; i++) {
    printf("%d: %f\n", i, out->data[i]);
    if (out->data[i] > out->data[best])
      best = i;
  }
  printf("Предсказание: %d\n", best);

  status = 0;

cleanup:
  if (f)
    fclose(f);
  for (int i = 0; i <= N_LAYERS; i++)
    if (act[i])
      tensor_free(act[i]);
  for (int i = 0; i < N_LAYERS; i++)
    if (layers[i])
      linear_free(layers[i]);
  return status;
}
