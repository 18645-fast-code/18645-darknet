#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <stdint.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define TOTAL_RUN 1000

static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

/**
 * If the input block has only space of 3x3
 */
void convolution_kernel_3_3(double *input, double *filter, double *output, int inputMatrixSize, int outputMatrixSize)
{
    // 16 registers
    __m256d f0, f1, f2;
    __m256d a0, a1, a2;
    __m256d r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

    f0 = _mm256_loadu_pd(filter);
    f1 = _mm256_loadu_pd(filter + 4);
    f2 = _mm256_loadu_pd(filter + 8);

    a0 = _mm256_loadu_pd(input);
    a1 = _mm256_loadu_pd(input + inputMatrixSize);
    a2 = _mm256_loadu_pd(input + 2 * inputMatrixSize);

    r0 = _mm256_setzero_pd();
    r1 = _mm256_setzero_pd();
    r2 = _mm256_setzero_pd();

    r0 = _mm256_fmadd_pd(a0, f0, r0);
    r1 = _mm256_fmadd_pd(a1, f1, r1);
    r1 = _mm256_fmadd_pd(a2, f2, r1);
    r0 = _mm256_add_pd(r0, r2);

    r0 = _mm256_hadd_pd(r0, r0);
    r1 = _mm256_hadd_pd(r1, r1);

    // Swap first two sum & second two sum
    r1 = _mm256_permute2f128_pd(r0, r0, 1 | (2 << 4));

    // Add together to get the sum of dot products
    r0 = _mm256_add_pd(r0, r1);

    output[0] = r0[0];
}

/**
 * If the input block has enough space of 4x3
 */
void convolution_kernel_4_3(double *input, double *filter, double *output, int inputMatrixSize, int outputMatrixSize)
{
    // 16 registers
    __m256d f0, f1, f2;
    __m256d a0, a1, a2;
    __m256d r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

    // Try to process a 4x3 block
    r0 = _mm256_setzero_pd();
    r1 = _mm256_setzero_pd();
    r2 = _mm256_setzero_pd();
    r3 = _mm256_setzero_pd();
    r4 = _mm256_setzero_pd();
    r5 = _mm256_setzero_pd();
    r6 = _mm256_setzero_pd();
    r7 = _mm256_setzero_pd();

    // 3 registers to load filter, each vector holds 4 values of filters (with one padded 0 at the end of each row)
    f0 = _mm256_loadu_pd(filter);
    f1 = _mm256_loadu_pd(filter + 4);
    f2 = _mm256_loadu_pd(filter + 8);

    a0 = _mm256_loadu_pd(input);
    a1 = _mm256_loadu_pd(input + inputMatrixSize);
    a2 = _mm256_loadu_pd(input + 2 * inputMatrixSize);

    r0 = _mm256_fmadd_pd(a0, f0, r0);
    r1 = _mm256_fmadd_pd(a1, f0, r1);
    r2 = _mm256_fmadd_pd(a1, f1, r2);
    r3 = _mm256_fmadd_pd(a2, f1, r3);

    // Reuse a0
    a0 = _mm256_loadu_pd(input + 3 * inputMatrixSize);

    // Now calculate the second part + third part
    r2 = _mm256_fmadd_pd(a2, f2, r2);
    r3 = _mm256_fmadd_pd(a0, f2, r3);

    // Ok now we aggregate the first part to sum(second, third)
    r0 = _mm256_add_pd(r0, r2);
    r1 = _mm256_add_pd(r1, r3);

    // Sum up each adjacent pair
    r0 = _mm256_hadd_pd(r0, r0);
    r1 = _mm256_hadd_pd(r1, r1);

    // Swap first two sum & second two sum
    r2 = _mm256_permute2f128_pd(r0, r0, 1 | (2 << 4));
    r3 = _mm256_permute2f128_pd(r1, r1, 1 | (2 << 4));

    // Add together to get the sum of dot products
    r0 = _mm256_add_pd(r0, r2);
    r1 = _mm256_add_pd(r1, r3);

    output[0] = r0[0];
    output[outputMatrixSize] = r1[0];
}

/**
 * Filter size 3x3.
 * We should pack the filter to be 3x4. (pad by 0) for easier vector process.
 * The kernel works on a 5x3 block of the input (vertical) and product a 3x1 output.
 * Kernel size 3.
 */
void convolution_kernel_5_3(double *input, double *filter, double *output, int inputMatrixSize, int outputMatrixSize)
{
    // 16 registers
    __m256d f0, f1, f2;
    __m256d a0, a1, a2;
    __m256d r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

    // Try to process a 5x3 block
    r0 = _mm256_setzero_pd();
    r1 = _mm256_setzero_pd();
    r2 = _mm256_setzero_pd();
    r3 = _mm256_setzero_pd();
    r4 = _mm256_setzero_pd();
    r5 = _mm256_setzero_pd();
    r6 = _mm256_setzero_pd();
    r7 = _mm256_setzero_pd();

    // 3 registers to load filter, each vector holds 4 values of filters (with one padded 0 at the end of each row)
    f0 = _mm256_loadu_pd(filter);
    f1 = _mm256_loadu_pd(filter + 4);
    f2 = _mm256_loadu_pd(filter + 8);

    a0 = _mm256_loadu_pd(input);
    a1 = _mm256_loadu_pd(input + inputMatrixSize);
    a2 = _mm256_loadu_pd(input + 2 * inputMatrixSize);

    r0 = _mm256_fmadd_pd(a0, f0, r0);
    r1 = _mm256_fmadd_pd(a1, f0, r1);
    r2 = _mm256_fmadd_pd(a2, f0, r2);
    r3 = _mm256_fmadd_pd(a1, f1, r3);
    r4 = _mm256_fmadd_pd(a2, f1, r4);

    // Reuse a0
    a0 = _mm256_loadu_pd(input + 3 * inputMatrixSize);
    // Reuse a1
    a1 = _mm256_loadu_pd(input + 4 * inputMatrixSize);
    r5 = _mm256_fmadd_pd(a0, f1, r5);

    // Now calculate the second part + third part
    r3 = _mm256_fmadd_pd(a2, f2, r3);
    r4 = _mm256_fmadd_pd(a0, f2, r4);
    r5 = _mm256_fmadd_pd(a2, f2, r5);

    // Ok now we aggregate the first part to sum(second, third)
    r0 = _mm256_add_pd(r0, r3);
    r1 = _mm256_add_pd(r2, r4);
    r2 = _mm256_add_pd(r3, r5);

    // Sum up each adjacent pair
    r0 = _mm256_hadd_pd(r0, r0);
    r1 = _mm256_hadd_pd(r1, r1);
    r2 = _mm256_hadd_pd(r2, r2);

    // Swap first two sum & second two sum
    r3 = _mm256_permute2f128_pd(r0, r0, 1 | (2 << 4));
    r4 = _mm256_permute2f128_pd(r1, r1, 1 | (2 << 4));
    r5 = _mm256_permute2f128_pd(r2, r2, 1 | (2 << 4));

    // Add together to get the sum of dot products
    r0 = _mm256_add_pd(r0, r3);
    r1 = _mm256_add_pd(r1, r4);
    r2 = _mm256_add_pd(r2, r5);

    output[0] = r0[0];
    output[outputMatrixSize] = r1[0];
    output[2 * outputMatrixSize] = r2[0];
}

/**
 * Filter size 3x3.
 */
void convolution_kernel(double *input, double *filter, double *output, int inputMatrixSize, int outputMatrixSize)
{
    for (int i = 0; i < inputMatrixSize; i += 3)
    {
        for (int j = 0; j < inputMatrixSize - 2; j += 1)
        {
            if (inputMatrixSize - i == 3)
            {
                convolution_kernel_3_3(input + (i * inputMatrixSize) + j, filter, output + (i * outputMatrixSize) + j, inputMatrixSize, outputMatrixSize);
            }
            else if (inputMatrixSize - i == 4)
            {
                convolution_kernel_4_3(input + (i * inputMatrixSize) + j, filter, output + (i * outputMatrixSize) + j, inputMatrixSize, outputMatrixSize);
            }
            else
            {
                convolution_kernel_5_3(input + (i * inputMatrixSize) + j, filter, output + (i * outputMatrixSize) + j, inputMatrixSize, outputMatrixSize);
            }
        }
    }
}

int main(int argc, char **argv)
{
    // Example input, filter, and output matrices
    int inputMatrixSize;
    if (argc == 2)
        inputMatrixSize = atoi(argv[1]);
    else
        inputMatrixSize = 10;

    int outputMatrixSize = inputMatrixSize - 2;
    int kernelSize = 3;

    printf("Driver initializing\n");

    double *input, *output, *filter;

    posix_memalign((void **)&input, 64, inputMatrixSize * inputMatrixSize * sizeof(double));
    // Pad the kernel (last column 0)
    posix_memalign((void **)&filter, 64, kernelSize * (kernelSize + 1) * sizeof(double));
    posix_memalign((void **)&output, 64, outputMatrixSize * outputMatrixSize * sizeof(double));

    // Randomly assign some number
    for (int i = 0; i < inputMatrixSize * inputMatrixSize; i++)
    {
        input[i] = ((double)rand()) / ((double)RAND_MAX);
    }

    for (int i = 0; i < outputMatrixSize * outputMatrixSize; i++)
    {
        output[i] = ((double)rand()) / ((double)RAND_MAX);
    }

    for (int i = 0; i < kernelSize; i++)
    {
        for (int j = 0; j < kernelSize + 1; j++)
        {
            int index = i * (kernelSize + 1) + j;
            if (j == kernelSize)
            {
                filter[index] = 0.0;
            }
            else
            {
                filter[i] = ((double)rand()) / ((double)RAND_MAX);
            }
        }
    }

    printf("Driver initialized\n");

    // Measure start time
    unsigned long long start = rdtsc();
    for (int i = 0; i < TOTAL_RUN; i++)
    {
        // Call the convolution_kernel function
        convolution_kernel(input, filter, output, inputMatrixSize, outputMatrixSize);
    }
    // Measure end time
    unsigned long long end = rdtsc();

    // Calculate elapsed cycles
    unsigned long long cycles = end - start;

    // Calculate FLOPs and FLOPs per cycle
    double flops = TOTAL_RUN * (outputMatrixSize * outputMatrixSize * 18);
    double flopsPerCycle = flops / cycles;

    printf("Elapsed cycles: %llu\n", cycles);
    printf("FLOPs per cycle: %f\n", flopsPerCycle);

    return 0;
}