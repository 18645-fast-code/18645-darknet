#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <stdint.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

unsigned long long st;
unsigned long long et;
unsigned long long float_instructions_executed = 0;

typedef struct
{
    int batch;
    int out_w;
    int out_h;
    int out_c;
    float *output;
} layerTest;

// Helper function to fill an array with random data
void fill_random(double *array, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = (double)rand() / RAND_MAX;
    }
}

void benchmark(layerTest l)
{
    // change activation
    float *a_avg = (float *)malloc(l.out_w * l.out_h * l.batch * sizeof(float));
    float *g = (float *)malloc(l.out_w * l.out_h * l.batch * sizeof(float));
    int alpha = 1;
    int b, w, h, c;
    st = rdtsc();
    for (b = 0; b < l.batch; ++b)
    {
        for (w = 0; w < l.out_w; w++)
        {
            for (h = 0; h < l.out_h; h++)
            {
                for (c = 0; c < l.out_c; c++)
                {
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
    free(a_avg); // Free allocated memory
    free(g);     // Free allocated memory
}

// Kernel size 4 * 8 with more interleaving
// This requires 8 output registers
void change_activation_kernel_v4(double *l_output, double *g, double *a_avg, double alpha)
{
    // 16 registers
    // 8 input registers
    // 1 temporary register to hold alpha
    // 8 output registers (reuse)
    __m256d r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16;
    r16 = _mm256_broadcast_sd(&alpha);

    // Row 1 & Row 2, LOAD
    // Row 1
    r15 = _mm256_loadu_pd(g);
    r14 = _mm256_loadu_pd(a_avg);
    r13 = _mm256_loadu_pd(g + 4);
    r12 = _mm256_loadu_pd(a_avg + 4);
    r1 = _mm256_loadu_pd(l_output);
    r2 = _mm256_loadu_pd(l_output + 4);

    // Row 2
    r11 = _mm256_loadu_pd(g + 8);
    r10 = _mm256_loadu_pd(a_avg + 8);
    r9 = _mm256_loadu_pd(g + 12);
    r8 = _mm256_loadu_pd(a_avg + 12);
    r3 = _mm256_loadu_pd(l_output + 8);
    r4 = _mm256_loadu_pd(l_output + 12);

    // Parallel all the muls
    r15 = _mm256_mul_pd(r15, r16);
    r13 = _mm256_mul_pd(r13, r16);
    r11 = _mm256_mul_pd(r11, r16);
    r9 = _mm256_mul_pd(r9, r16);

    // Do FMAs
    r1 = _mm256_fmadd_pd(r15, r14, r1);
    r2 = _mm256_fmadd_pd(r13, r12, r2);
    r3 = _mm256_fmadd_pd(r11, r10, r3);
    r4 = _mm256_fmadd_pd(r9, r8, r4);

    // Row 3 & Row 4, LOAD
    // Row 3
    r15 = _mm256_loadu_pd(g + 16);
    r14 = _mm256_loadu_pd(a_avg + 16);
    r13 = _mm256_loadu_pd(g + 20);
    r12 = _mm256_loadu_pd(a_avg + 20);
    r5 = _mm256_loadu_pd(l_output + 16);
    r6 = _mm256_loadu_pd(l_output + 20);

    // Row 4
    r11 = _mm256_loadu_pd(g + 24);
    r10 = _mm256_loadu_pd(a_avg + 24);
    r9 = _mm256_loadu_pd(g + 28);
    r8 = _mm256_loadu_pd(a_avg + 28);
    r7 = _mm256_loadu_pd(l_output + 24);
    r8 = _mm256_loadu_pd(l_output + 28);

    // Parallel all the muls
    r15 = _mm256_mul_pd(r15, r16);
    r13 = _mm256_mul_pd(r13, r16);
    r11 = _mm256_mul_pd(r11, r16);
    r9 = _mm256_mul_pd(r9, r16);

    // FMAs
    r5 = _mm256_fmadd_pd(r15, r14, r5);
    r6 = _mm256_fmadd_pd(r15, r14, r6);
    r7 = _mm256_fmadd_pd(r13, r12, r7);
    r8 = _mm256_fmadd_pd(r11, r10, r8);
}

// Kernel size 8 * 8
void change_activation_kernel_v3(double *l_output, double *g, double *a_avg, double alpha)
{
    // 16 output Registers
    __m256d r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16;
    r16 = _mm256_broadcast_sd(&alpha);
    // Row 1 - Load
    r15 = _mm256_loadu_pd(g);
    r14 = _mm256_loadu_pd(a_avg);
    r13 = _mm256_loadu_pd(g + 4);
    r12 = _mm256_loadu_pd(a_avg + 4);
    // Row 1 - Compute
    r1 = _mm256_loadu_pd(l_output);
    r2 = _mm256_loadu_pd(l_output + 4);
    r15 = _mm256_mul_pd(r15, r16);
    r1 = _mm256_fmadd_pd(r15, r14, r1);
    r13 = _mm256_mul_pd(r13, r16);
    r2 = _mm256_fmadd_pd(r13, r12, r2);
    // Row 2 - Load
    r11 = _mm256_loadu_pd(g + 8);
    r10 = _mm256_loadu_pd(a_avg + 8);
    r9 = _mm256_loadu_pd(g + 12);
    r8 = _mm256_loadu_pd(a_avg + 12);
    // Row 1 - Store
    _mm256_storeu_pd(l_output, r1);
    _mm256_storeu_pd(l_output + 4, r2);
    // Row 2 - Compute
    r3 = _mm256_loadu_pd(l_output + 8);
    r4 = _mm256_loadu_pd(l_output + 12);
    r11 = _mm256_mul_pd(r11, r16);
    r3 = _mm256_fmadd_pd(r11, r10, r3);
    r9 = _mm256_mul_pd(r9, r16);
    r4 = _mm256_fmadd_pd(r9, r8, r4);
    // Row 3 - Load
    r15 = _mm256_loadu_pd(g + 16);
    r14 = _mm256_loadu_pd(a_avg + 16);
    r13 = _mm256_loadu_pd(g + 20);
    r12 = _mm256_loadu_pd(a_avg + 20);
    // Row 2 - Store
    _mm256_storeu_pd(l_output + 8, r3);
    _mm256_storeu_pd(l_output + 12, r4);
    // Row 3 - Compute
    r5 = _mm256_loadu_pd(l_output + 16);
    r6 = _mm256_loadu_pd(l_output + 20);
    r15 = _mm256_mul_pd(r15, r16);
    r5 = _mm256_fmadd_pd(r15, r14, r5);
    r13 = _mm256_mul_pd(r13, r16);
    r6 = _mm256_fmadd_pd(r13, r12, r6);
    // Row 4 - Load
    r11 = _mm256_loadu_pd(g + 24);
    r10 = _mm256_loadu_pd(a_avg + 24);
    r9 = _mm256_loadu_pd(g + 28);
    r8 = _mm256_loadu_pd(a_avg + 28);
    // Row 3 - Store
    _mm256_storeu_pd(l_output + 16, r5);
    _mm256_storeu_pd(l_output + 20, r6);
    // Row 4 - Compute
    r7 = _mm256_loadu_pd(l_output + 24);
    r8 = _mm256_loadu_pd(l_output + 28);
    r11 = _mm256_mul_pd(r11, r16);
    r7 = _mm256_fmadd_pd(r11, r10, r7);
    r9 = _mm256_mul_pd(r9, r16);
    r8 = _mm256_fmadd_pd(r9, r8, r8);

    // Row 5 - Load
    r15 = _mm256_loadu_pd(g + 32);
    r14 = _mm256_loadu_pd(a_avg + 32);
    r13 = _mm256_loadu_pd(g + 36);
    r12 = _mm256_loadu_pd(a_avg + 36);
    // Row 4 - Store
    _mm256_storeu_pd(l_output + 24, r7);
    _mm256_storeu_pd(l_output + 28, r8);
    // Row 5 - Compute
    r9 = _mm256_loadu_pd(l_output + 32);
    r10 = _mm256_loadu_pd(l_output + 36);
    r15 = _mm256_mul_pd(r15, r16);
    r9 = _mm256_fmadd_pd(r15, r14, r9);
    r13 = _mm256_mul_pd(r13, r16);
    r10 = _mm256_fmadd_pd(r13, r12, r10);
    // Row 6 - Load
    r15 = _mm256_loadu_pd(g + 40);
    r14 = _mm256_loadu_pd(a_avg + 40);
    r13 = _mm256_loadu_pd(g + 44);
    r12 = _mm256_loadu_pd(a_avg + 44);
    // Row 5 - Store
    _mm256_storeu_pd(l_output + 32, r9);
    _mm256_storeu_pd(l_output + 36, r10);
    // Row 6 - Compute
    r11 = _mm256_loadu_pd(l_output + 40);
    r12 = _mm256_loadu_pd(l_output + 44);
    r15 = _mm256_mul_pd(r15, r16);
    r11 = _mm256_fmadd_pd(r15, r14, r11);
    r13 = _mm256_mul_pd(r13, r16);
    r12 = _mm256_fmadd_pd(r13, r12, r12);
    // Row 7 - Load
    r15 = _mm256_loadu_pd(g + 48);
    r14 = _mm256_loadu_pd(a_avg + 48);
    // Row 6 - Store
    _mm256_storeu_pd(l_output + 40, r11);
    _mm256_storeu_pd(l_output + 44, r12);
    // Row 7 - Compute
    r13 = _mm256_loadu_pd(l_output + 48);
    r15 = _mm256_mul_pd(r15, r16);
    r13 = _mm256_fmadd_pd(r15, r14, r13);
    r15 = _mm256_loadu_pd(g + 52);
    r15 = _mm256_mul_pd(r15, r16);
    r14 = _mm256_loadu_pd(a_avg + 52);
    r16 = _mm256_loadu_pd(l_output + 52);
    r16 = _mm256_fmadd_pd(r15, r14, r16);
    // Row 8 - Load
    r15 = _mm256_loadu_pd(g + 56);
    r14 = _mm256_loadu_pd(a_avg + 56);
    r13 = _mm256_loadu_pd(g + 60);
    r12 = _mm256_loadu_pd(a_avg + 60);
    // Row 7 - Store
    _mm256_storeu_pd(l_output + 48, r13);
    _mm256_storeu_pd(l_output + 52, r16);
    // Row 8 - Compute
    r1 = _mm256_loadu_pd(l_output + 56);
    r2 = _mm256_loadu_pd(l_output + 60);
    r15 = _mm256_mul_pd(r15, r16);
    r1 = _mm256_fmadd_pd(r15, r14, r1);
    r13 = _mm256_mul_pd(r13, r16);
    r2 = _mm256_fmadd_pd(r13, r12, r2);
    // Row 8 - Store
    _mm256_storeu_pd(l_output + 56, r1);
    _mm256_storeu_pd(l_output + 60, r2);
}

// Kernel size 7 * 8
void change_activation_kernel_v2(double *l_output, double *g, double *a_avg, double alpha)
{
    // 14 output registers
    __m256d r1, r2, r3, r4, r5, r6, r7;
    __m256d r8, r9, r10, r11, r12, r13, r14, r15, r16;

    r16 = _mm256_broadcast_sd(&alpha);

    // Row 1
    r15 = _mm256_loadu_pd(g);
    r14 = _mm256_loadu_pd(a_avg);
    r13 = _mm256_loadu_pd(g + 4);
    r12 = _mm256_loadu_pd(a_avg + 4);
    r1 = _mm256_loadu_pd(l_output);
    r2 = _mm256_loadu_pd(l_output + 4);
    r15 = _mm256_mul_pd(r15, r16);
    r1 = _mm256_fmadd_pd(r15, r14, r1);
    r13 = _mm256_mul_pd(r13, r16);
    r2 = _mm256_fmadd_pd(r13, r12, r2);

    // Row 2
    r11 = _mm256_loadu_pd(g + 8);
    r10 = _mm256_loadu_pd(a_avg + 8);
    r9 = _mm256_loadu_pd(g + 12);
    r8 = _mm256_loadu_pd(a_avg + 12);
    r3 = _mm256_loadu_pd(l_output + 8);
    r4 = _mm256_loadu_pd(l_output + 12);
    r11 = _mm256_mul_pd(r11, r16);
    r3 = _mm256_fmadd_pd(r11, r10, r3);
    r9 = _mm256_mul_pd(r9, r16);
    r4 = _mm256_fmadd_pd(r9, r8, r4);

    // Row 3
    r15 = _mm256_loadu_pd(g + 16);
    r14 = _mm256_loadu_pd(a_avg + 16);
    r13 = _mm256_loadu_pd(g + 20);
    r12 = _mm256_loadu_pd(a_avg + 20);
    r5 = _mm256_loadu_pd(l_output + 16);
    r6 = _mm256_loadu_pd(l_output + 20);
    r15 = _mm256_mul_pd(r15, r16);
    r5 = _mm256_fmadd_pd(r15, r14, r5);
    r13 = _mm256_mul_pd(r13, r16);
    r6 = _mm256_fmadd_pd(r13, r12, r6);

    // Row 4
    r11 = _mm256_loadu_pd(g + 24);
    r10 = _mm256_loadu_pd(a_avg + 24);
    r9 = _mm256_loadu_pd(g + 28);
    r8 = _mm256_loadu_pd(a_avg + 28);
    r7 = _mm256_loadu_pd(l_output + 24);
    r11 = _mm256_mul_pd(r11, r16);
    r7 = _mm256_fmadd_pd(r11, r10, r7);
    r8 = _mm256_loadu_pd(l_output + 28);
    r9 = _mm256_mul_pd(r9, r8);
    r8 = _mm256_fmadd_pd(r9, r16, r8); // Fix r8, r8 cannot be reused

    // Row 5
    r15 = _mm256_loadu_pd(g + 32);
    r14 = _mm256_loadu_pd(a_avg + 32);
    r13 = _mm256_loadu_pd(g + 36);
    r12 = _mm256_loadu_pd(a_avg + 36);
    r9 = _mm256_loadu_pd(l_output + 32);
    r10 = _mm256_loadu_pd(l_output + 36);
    r15 = _mm256_mul_pd(r15, r16);
    r9 = _mm256_fmadd_pd(r15, r14, r9); // Fix r9, r9 cannot be reused
    r13 = _mm256_mul_pd(r13, r16);
    r10 = _mm256_fmadd_pd(r13, r12, r10); // Fix r10, r10 cannot be reused

    // Row 6
    r15 = _mm256_loadu_pd(g + 40);
    r14 = _mm256_loadu_pd(a_avg + 40);
    r13 = _mm256_loadu_pd(g + 44);
    r12 = _mm256_loadu_pd(a_avg + 44);
    r11 = _mm256_loadu_pd(l_output + 40);
    r15 = _mm256_mul_pd(r15, r16);
    r11 = _mm256_fmadd_pd(r15, r14, r11); // Fix r11, r11 cannot be reused
    r13 = _mm256_mul_pd(r13, r12);
    r12 = _mm256_loadu_pd(l_output + 44);
    r12 = _mm256_fmadd_pd(r13, r16, r12); // Fix r12, r12 cannot be reused

    // Row 7
    r15 = _mm256_loadu_pd(g + 48);
    r14 = _mm256_loadu_pd(a_avg + 48);
    r13 = _mm256_loadu_pd(l_output + 48);
    r15 = _mm256_mul_pd(r15, r16);
    r13 = _mm256_fmadd_pd(r15, r14, r13); // Fix r13, r13 cannot be reusued
    r15 = _mm256_loadu_pd(g + 52);
    r15 = _mm256_mul_pd(r15, r16);
    r16 = _mm256_loadu_pd(a_avg + 48);
    r14 = _mm256_loadu_pd(l_output + 52);
    r14 = _mm256_fmadd_pd(r15, r16, r14);

    // Store
    _mm256_storeu_pd(l_output, r1);
    _mm256_storeu_pd(l_output + 4, r2);
    _mm256_storeu_pd(l_output + 8, r3);
    _mm256_storeu_pd(l_output + 12, r4);
    _mm256_storeu_pd(l_output + 16, r5);
    _mm256_storeu_pd(l_output + 20, r6);
    _mm256_storeu_pd(l_output + 24, r7);
    _mm256_storeu_pd(l_output + 28, r8);
    _mm256_storeu_pd(l_output + 32, r9);
    _mm256_storeu_pd(l_output + 36, r10);
    _mm256_storeu_pd(l_output + 40, r11);
    _mm256_storeu_pd(l_output + 44, r12);
    _mm256_storeu_pd(l_output + 48, r13);
    _mm256_storeu_pd(l_output + 52, r14);
}

// Kernel size 6 * 8
void change_activation_kernel(double *l_output, double *g, double *a_avg, double alpha)
{
    // 12 output registers
    __m256d r1, r2, r3, r4, r5, r6, r7;
    __m256d r8, r9, r10, r11, r12, r13, r14, r15, r16;

    r16 = _mm256_broadcast_sd(&alpha);

    // Row 1
    r15 = _mm256_loadu_pd(g);
    r14 = _mm256_loadu_pd(a_avg);
    r13 = _mm256_loadu_pd(g + 4);
    r12 = _mm256_loadu_pd(a_avg + 4);

    r1 = _mm256_loadu_pd(l_output);
    r2 = _mm256_loadu_pd(l_output + 4);

    r15 = _mm256_mul_pd(r15, r16);
    r1 = _mm256_fmadd_pd(r15, r14, r1);

    r13 = _mm256_mul_pd(r13, r16);
    r2 = _mm256_fmadd_pd(r13, r12, r2);

    // Row 2
    r11 = _mm256_loadu_pd(g + 8);
    r10 = _mm256_loadu_pd(a_avg + 8);
    r9 = _mm256_loadu_pd(g + 12);
    r8 = _mm256_loadu_pd(a_avg + 12);

    r3 = _mm256_loadu_pd(l_output + 8);
    r4 = _mm256_loadu_pd(l_output + 12);

    r11 = _mm256_mul_pd(r11, r16);
    r3 = _mm256_fmadd_pd(r11, r10, r3);

    r9 = _mm256_mul_pd(r9, r16);
    r4 = _mm256_fmadd_pd(r9, r8, r4);

    // Row 3
    r15 = _mm256_loadu_pd(g + 16);
    r14 = _mm256_loadu_pd(a_avg + 16);
    r13 = _mm256_loadu_pd(g + 20);
    r12 = _mm256_loadu_pd(a_avg + 20);

    r5 = _mm256_loadu_pd(l_output + 16);
    r6 = _mm256_loadu_pd(l_output + 20);

    r15 = _mm256_mul_pd(r15, r16);
    r5 = _mm256_fmadd_pd(r15, r14, r5);

    r13 = _mm256_mul_pd(r13, r16);
    r6 = _mm256_fmadd_pd(r13, r12, r6);

    // Row 4
    r11 = _mm256_loadu_pd(g + 24);
    r10 = _mm256_loadu_pd(a_avg + 24);
    r9 = _mm256_loadu_pd(g + 28);
    r8 = _mm256_loadu_pd(a_avg + 28);

    r7 = _mm256_loadu_pd(l_output + 24);

    r11 = _mm256_mul_pd(r11, r16);
    r7 = _mm256_fmadd_pd(r11, r10, r7);

    r9 = _mm256_mul_pd(r9, r8);
    r8 = _mm256_loadu_pd(l_output + 28);
    // Fix r8, r8 cannot be reused
    r8 = _mm256_fmadd_pd(r9, r16, r8);

    // Row 5
    r15 = _mm256_loadu_pd(g + 32);
    r14 = _mm256_loadu_pd(a_avg + 32);
    r13 = _mm256_loadu_pd(g + 36);
    r12 = _mm256_loadu_pd(a_avg + 36);

    r9 = _mm256_loadu_pd(l_output + 32);
    r10 = _mm256_loadu_pd(l_output + 36);

    r15 = _mm256_mul_pd(r15, r16);
    // Fix r9, r9 cannot be reused
    r9 = _mm256_fmadd_pd(r15, r14, r9);

    r13 = _mm256_mul_pd(r13, r16);
    // Fix r10, r10 cannot be reused
    r10 = _mm256_fmadd_pd(r13, r12, r10);

    // Row 6
    r15 = _mm256_loadu_pd(g + 40);
    r14 = _mm256_loadu_pd(a_avg + 40);
    r13 = _mm256_loadu_pd(g + 44);

    r11 = _mm256_loadu_pd(l_output + 40);

    r15 = _mm256_mul_pd(r15, r16);
    // Fix r11, r11 cannot be reused
    r11 = _mm256_fmadd_pd(r15, r14, r11);

    r13 = _mm256_mul_pd(r13, r12);
    r12 = _mm256_loadu_pd(l_output + 44);
    // Fix r12, r12 cannot be reused
    r12 = _mm256_fmadd_pd(r13, r16, r12);

    // Store
    _mm256_storeu_pd(l_output, r1);
    _mm256_storeu_pd(l_output + 4, r2);
    _mm256_storeu_pd(l_output + 8, r3);
    _mm256_storeu_pd(l_output + 12, r4);
    _mm256_storeu_pd(l_output + 16, r5);
    _mm256_storeu_pd(l_output + 20, r6);
    _mm256_storeu_pd(l_output + 24, r7);
    _mm256_storeu_pd(l_output + 28, r8);
    _mm256_storeu_pd(l_output + 32, r9);
    _mm256_storeu_pd(l_output + 36, r10);
    _mm256_storeu_pd(l_output + 40, r11);
    _mm256_storeu_pd(l_output + 44, r12);
}

void benchmark_kernel1()
{
    // Dimensions of the kernel
    const int width = 8;
    const int height = 8;
    const int size = width * height;
    unsigned long long t0, t1, t_total;

    // Allocate memory for arrays
    double *l_output = (double *)malloc(sizeof(double) * size);
    double *g = (double *)malloc(sizeof(double) * size);
    double *a_avg = (double *)malloc(sizeof(double) * size);
    double alpha = 0.5;
    // Fill the g and a_avg arrays with random data
    fill_random(g, size);
    fill_random(a_avg, size);

    // Number of iterations for benchmarking
    const int iterations = 1000000;

    // Call the kernel multiple times
    for (int i = 0; i < iterations; ++i)
    {
        t0 = rdtsc();
        change_activation_kernel_v4(l_output, g, a_avg, alpha);
        t1 = rdtsc();
        t_total += (t1 - t0);
    }

    double total_flops = 3.0 * size * iterations;
    printf("performance is %lf FLOPS per cycle\n", total_flops / ((double)(t_total)));

    free(l_output);
    free(g);
    free(a_avg);
}

int main_old()
{
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
}

int main()
{
    benchmark_kernel1();
    return 0;
}
