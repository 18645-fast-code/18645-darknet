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

/**
 * Filter size 3x3.
 * We should pack the filter to be 3x4. (pad by 0) for easier vector process.
 * The kernel works on a 5x3 block of the input (vertical) and product a 3x1 output.
 * Kernel size 3.
 */
void convolution_kernel(double *input, double *filter, double *output, int inputMatrixSize, int outputMatrixSize)
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

    output[0] = r1[0];
    output[outputMatrixSize] = r2[0];
    output[2 * outputMatrixSize] = r3[0];
    output[3 * outputMatrixSize] = r4[0];
}

int main() {
    // Example input, filter, and output matrices
    int inputMatrixSize = 5;  // Example size, adjust according to your requirements
    int outputMatrixSize = 3; // Example size, adjust according to your requirements

    double input[] = {
        // Example input matrix (5x3)
        // Replace with your actual data
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0
    };

    double filter[] = {
        // Example filter matrix (3x3)
        // Replace with your actual filter data
        0.1, 0.2, 0.3, 
        0.5, 0.6, 0.7, 
        0.9, 1.0, 1.1
    };

    double output[outputMatrixSize * outputMatrixSize];

    // Measure start time
    unsigned long long start = rdtsc();

    // Call the convolution_kernel function
    convolution_kernel(input, filter, output, inputMatrixSize, outputMatrixSize);

    // Measure end time
    unsigned long long end = rdtsc();

    // Calculate elapsed cycles
    unsigned long long cycles = end - start;

    // Print the result
    for (int i = 0; i < outputMatrixSize; ++i) {
        for (int j = 0; j < outputMatrixSize; ++j) {
            printf("%f ", output[i * outputMatrixSize + j]);
        }
        printf("\n");
    }

    // Calculate FLOPs and FLOPs per cycle
    double flops = 2.0 * inputMatrixSize * inputMatrixSize * inputMatrixSize * outputMatrixSize * outputMatrixSize; // Assuming 2 FLOPs per multiplication-addition
    double frequency = MAX_FREQ;
    double seconds = (double)cycles / frequency;
    double flopsPerCycle = flops / cycles;

    printf("Elapsed cycles: %llu\n", cycles);
    printf("Elapsed time: %f seconds\n", seconds);
    printf("FLOPs: %f\n", flops);
    printf("FLOPs per cycle: %f\n", flopsPerCycle);

    return 0;
}