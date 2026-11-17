#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arch.h"
#include "dataset.h"

#define TRAIN_MAX 60000
#define TEST_MAX 10000
#define EPOCHS 10
#define BATCH 32
#define ALPHA 0.05f
#define CLASSES 10
#define LOG_EVERY 50  // писать строку в лог каждые N батчей
#define QUICK_TEST 1000 // на скольких тестовых примерах считать быструю точность

static float accuracy(Arch *net, const MnistSample *data, int n) {
  int correct = 0;
  for (int i = 0; i < n; i++) {
    memcpy(net->act[0]->data, data[i].pixels, sizeof(data[i].pixels));
    Tensor *out = arch_forward(net);
    int best = 0;
    for (int k = 1; k < CLASSES; k++)
      if (out->data[k] > out->data[best])
        best = k;
    if (best == data[i].label)
      correct++;
  }
  return (float)correct / n;
}

static void log_row(FILE *f, long step, int epoch, double loss, float acc) {
  fprintf(f, "%ld,%d,%f,%f\n", step, epoch, loss, acc);
  fflush(f);
}

int main(int argc, char **argv) {
  const char *train_path = argc > 1 ? argv[1] : "mnist_train.csv";
  const char *test_path = argc > 2 ? argv[2] : "mnist_test.csv";
  srand(42);

  MnistSample *train = malloc(sizeof(MnistSample) * TRAIN_MAX);
  MnistSample *test = malloc(sizeof(MnistSample) * TEST_MAX);
  int *idx = malloc(sizeof(int) * TRAIN_MAX);
  if (!train || !test || !idx)
    return 1;

  int n_train = load_mnist(train, train_path, TRAIN_MAX);
  int n_test = load_mnist(test, test_path, TEST_MAX);
  if (n_train <= 0 || n_test <= 0)
    return 1;

  int sizes[] = {NUM_PIXELS, 128, 64, 32, CLASSES};
  Arch *net = arch_new(5, sizes);
  if (!net)
    return 1;
  arch_init_random(net);

  FILE *log = fopen("training.csv", "w");
  if (!log)
    return 1;
  fprintf(log, "step,epoch,train_loss,test_acc\n");
  fflush(log);

  for (int i = 0; i < n_train; i++)
    idx[i] = i;

  float target[CLASSES];
  Tensor *dy = net->grad[net->size - 1];
  long step = 0;
  double win_loss = 0;
  int win_n = 0;

  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    shuffle_indices(idx, n_train);
    double loss_sum = 0;

    for (int start = 0; start < n_train; start += BATCH) {
      int cnt = n_train - start < BATCH ? n_train - start : BATCH;

      for (int k = 0; k < cnt; k++) {
        const MnistSample *s = &train[idx[start + k]];

        memcpy(net->act[0]->data, s->pixels, sizeof(s->pixels));
        Tensor *out = arch_forward(net);

        memset(target, 0, sizeof(target));
        target[s->label] = 1.0f;

        double l = loss_MSE(out->data, target, CLASSES);
        loss_sum += l;
        win_loss += l;
        win_n++;
        loss_MSE_grad(out->data, target, dy->data, CLASSES);
        arch_backward(net);
      }
      arch_step(net, ALPHA, cnt);

      step++;
      if (step % LOG_EVERY == 0) {
        int n_quick = n_test < QUICK_TEST ? n_test : QUICK_TEST;
        log_row(log, step, epoch, win_loss / win_n, accuracy(net, test, n_quick));
        win_loss = 0;
        win_n = 0;
      }
    }

    float acc = accuracy(net, test, n_test);
    printf("epoch %2d  loss %.5f  acc %.4f\n", epoch, loss_sum / n_train, acc);
    if (win_n > 0) {
      log_row(log, step, epoch, win_loss / win_n, acc);
      win_loss = 0;
      win_n = 0;
    }
  }

  fclose(log);
  arch_free(net);
  free(idx);
  free(test);
  free(train);
  return 0;
}
