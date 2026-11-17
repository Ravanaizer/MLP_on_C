#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"

#define MAX_LINE_LENGTH 8192

int load_mnist(MnistSample *dataset, const char *filename, int max_samples) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror(filename);
    return -1;
  }

  char line[MAX_LINE_LENGTH];
  int count = 0;

  while (count < max_samples && fgets(line, sizeof(line), file)) {
    char *ptr = line;
    char *endptr;

    long label = strtol(ptr, &endptr, 10);
    if (ptr == endptr)
      continue;

    memset(&dataset[count], 0, sizeof(MnistSample));
    dataset[count].label = (int)label;

    for (int i = 0; i < NUM_PIXELS; i++) {
      if (*endptr != ',')
        break;
      ptr = endptr + 1;
      long pixel_val = strtol(ptr, &endptr, 10);
      dataset[count].pixels[i] = (float)pixel_val / 255.0f;
    }
    count++;
  }

  fclose(file);
  return count;
}

void shuffle_indices(int *idx, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
}
