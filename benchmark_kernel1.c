#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <stdint.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

unsigned long long st;
unsigned long long et;
unsigned long long float_instructions_executed = 0;

typedef struct {
    int batch;
    int out_w;
    int out_h;
    int out_c;
    float *output;
} layerTest;


void benchmark(layerTest l){
    // change activation
    float *a_avg = (float *)malloc(l.out_w * l.out_h * l.batch * sizeof(float));
    float *g = (float *)malloc(l.out_w * l.out_h * l.batch * sizeof(float));
    int alpha = 1;
    int b, w, h, c;
    st = rdtsc();
    for (b = 0; b < l.batch; ++b) {
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++) {
                    l.output[w + l.out_w * (h + l.out_h * (c + l.out_c * b))] +=
                        alpha *
                        g[w + l.out_w * (h + l.out_h * b)] *
                        a_avg[w + l.out_w * (h + l.out_h * b)];
                    float_instructions_executed += 1;
                }
            }
        }
    }
    et = rdtsc();
    free(a_avg);  // Free allocated memory
    free(g);      // Free allocated memory
}

int main() {
    layerTest l;
    l.batch = 1;
    l.out_w = 10;
    l.out_h = 10;
    l.out_c = 1;

    // Allocate memory for l.output using malloc
    l.output = (float *)malloc(l.batch * l.out_w * l.out_h * l.out_c * sizeof(float));

    int run_multiplier = 0;
    run_multiplier = l.batch * l.out_w * l.out_c * l.out_h;
    benchmark(l);
    // Calculate OPI (Operations Per Instruction)
    double OPI = (double)run_multiplier / float_instructions_executed;
    // Calculate OPC (Operations Per Cycle)
    double OPC = OPI / ((et - st) / MAX_FREQ * BASE_FREQ);
    // printf("#Operations/cycle: %lf\n\r", (run_multiplier) / ((double)(et - st) / MAX_FREQ / BASE_FREQ));
    printf("#Operations/Instruction: %lf\n", OPI);
    printf("#Operations/cycle: %lf\n", OPC);

    // Free allocated memory for l.output
    free(l.output);

    return 0;
}
    

