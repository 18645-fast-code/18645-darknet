#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <stdint.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

// Helper function to fill an array with random data
void fill_random(double *array, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = (double)rand() / RAND_MAX;
    }
}

void change_activation_kernel(double *l_output, double *g, double *a_avg, double alpha)
{
    // 16 registers
    // 8 output registers
    __m256d r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16;
    r16 = _mm256_broadcast_sd(&alpha);

    r15 = _mm256_loadu_pd(g);
    r14 = _mm256_loadu_pd(g + 4);
    r13 = _mm256_loadu_pd(g + 8);
    r12 = _mm256_loadu_pd(g + 12);
    r11 = _mm256_loadu_pd(g + 16);
    r10 = _mm256_loadu_pd(g + 20);

    r15 = _mm256_mul_pd(r15, r16);
    r14 = _mm256_mul_pd(r14, r16);
    r13 = _mm256_mul_pd(r13, r16);
    r12 = _mm256_mul_pd(r12, r16);
    r11 = _mm256_mul_pd(r11, r16);
    r10 = _mm256_mul_pd(r10, r16);

    r9 = _mm256_loadu_pd(a_avg);
    r1 = _mm256_loadu_pd(l_output);
    r8 = _mm256_loadu_pd(a_avg + 4);
    r2 = _mm256_loadu_pd(l_output + 4);
    r7 = _mm256_loadu_pd(a_avg + 8);
    r3 = _mm256_loadu_pd(l_output + 8);
    r6 =  _mm256_loadu_pd(a_avg + 12);
    r4 =  _mm256_loadu_pd(l_output + 12);

    r1 = _mm256_fmadd_pd(r15, r9, r1);
    r2 = _mm256_fmadd_pd(r14, r8, r2);
    r3 = _mm256_fmadd_pd(r13, r7, r3);
    r4 = _mm256_fmadd_pd(r12, r6, r4);

    r15 = _mm256_loadu_pd(a_avg + 16);
    r5 = _mm256_loadu_pd(l_output + 16);
    r14 = _mm256_loadu_pd(a_avg + 20);
    r6 = _mm256_loadu_pd(l_output + 20);
    r13 = _mm256_loadu_pd(g + 24);
    r12 = _mm256_loadu_pd(g + 28);

    r13 = _mm256_mul_pd(r13, r16);
    r12 = _mm256_mul_pd(r12, r16);
    
    r5 = _mm256_fmadd_pd(r11, r15, r5);
    r6 = _mm256_fmadd_pd(r10, r14, r6);

    r11 = _mm256_loadu_pd(a_avg + 24);
    r7 = _mm256_loadu_pd(l_output + 24);
    r10 = _mm256_loadu_pd(a_avg + 28);
    r8 = _mm256_loadu_pd(l_output + 28);

    r7 = _mm256_fmadd_pd(r13, r11, r7);
    r8 = _mm256_fmadd_pd(r12, r10, r8);

    // Store
    _mm256_storeu_pd(l_output, r1);
    _mm256_storeu_pd(l_output + 4, r2);
    _mm256_storeu_pd(l_output + 8, r3);
    _mm256_storeu_pd(l_output + 12, r4);
    _mm256_storeu_pd(l_output + 16, r5);
    _mm256_storeu_pd(l_output + 20, r6);
    _mm256_storeu_pd(l_output + 24, r7);
    _mm256_storeu_pd(l_output + 28, r8);
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
        change_activation_kernel_v3(l_output, g, a_avg, alpha);
        t1 = rdtsc();
        t_total += (t1 - t0);
    }

    double total_flops = 2.0 * size * iterations;
    printf("performance is %lf FLOPS per cycle\n", total_flops / ((double)(t_total)));

    free(l_output);
    free(g);
    free(a_avg);
}

int main()
{
    benchmark_kernel1();
    return 0;
}
