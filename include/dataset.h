#ifndef DATASET_H
#define DATASET_H

#define NUM_PIXELS 784

typedef struct {
  int label;
  float pixels[NUM_PIXELS];
} MnistSample;

// Возвращает число реально загруженных примеров, -1 при ошибке открытия
int load_mnist(MnistSample *dataset, const char *filename, int max_samples);

// Перемешивает массив индексов (Fisher-Yates), сами примеры не копируются
void shuffle_indices(int *idx, int n);

#endif
