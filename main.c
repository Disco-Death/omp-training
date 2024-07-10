#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <string.h>

#ifdef _OPENMP
// omp_set_nested(1);
// here omp_get_num_procs implementation exists
#else
int omp_get_num_procs() { return 1; }
double omp_get_wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}
#endif

#define VER_X
#undef VER_X

#ifndef RAND_MAX
#define RAND_MAX ((int)((unsigned)~0 >> 1))
#endif

/* A = 11 * 5 * 10 = 550 */
#define A 550

#define IS_WHOLE_PART_EVEN(num) (((((int)(num)) % 2) == 0))
#define SET_SEED_POINT(pSeed, seed) ((pSeed) = (seed))

#define RANDOM_DOUBLE(low, high, pSeed) ((low) + ((double)rand_r(pSeed) / ((double)(RAND_MAX) + 1)) * ((high) - (low)))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void insertion_sort(double *array, unsigned int size)
{
    unsigned int i, j;
    double x;

    for (i = 1; i < size; i++)
    {
        x = array[i];
        j = i;

        while (j > 0 && array[j - 1] > x)
        {
            array[j] = array[j - 1];
            j -= 1;
        }

        array[j] = x;
    }
}

void merge_sort(double *src1, int n1, double *src2, int n2, double *dst)
{
    int i = 0, i1 = 0, i2 = 0;
    while (i < n1 + n2)
        dst[i++] = src1[i1] > src2[i2] && i2 < n2 ? src2[i2++] : src1[i1++];
}

static inline void sort(double *array, int n, double *dst)
{
    int n1 = n / 2;
    int n2 = n - n1;
#pragma omp sections
    {
#pragma omp section
        {
            insertion_sort(array, n1);
        }
#pragma omp section
        {
            insertion_sort(array + n1, n2);
        }
    }
#pragma omp single
    merge_sort(array, n1, array + n1, n2, dst);
}

/* X(B) = 1 + ((550 mod 47) mod B) = 1 + (33 mod B)
 * X(7) = 1 + (33 mod 7) = 6
 * X(8) = 1 + (33 mod 8) = 2
 * X(6) = 1 + (33 mod 6) = 4 */

int main(int argc, char *argv[])
{
    unsigned int i, j, pSeed, N, N2, finished = 0;
    double X, minimum;
    double *restrict M1, *restrict M2, *restrict M2Copy;
    double T1, T2;

    N = atoi(argv[1]); // N равен первому параметру командной строки
    const int M = atoi(argv[2]);

    N2 = N / 2;
    M2Copy = (double *)malloc((N2 + 1) * sizeof(double));
    M1 = (double *)malloc(N * sizeof(double));
    M2 = (double *)malloc(N2 * sizeof(double));
    M2Copy[0] = 0.0;

    T1 = omp_get_wtime(); // запомнить текущее время T1 */

#pragma omp parallel sections num_threads(2) shared(i, finished)
    {
#ifdef _OPENMP
#pragma omp section
        {
            double time = 0;
            while (!finished)
            {
                double time_temp = omp_get_wtime();
                if (time_temp - time < 1)
                {
                    usleep(100);
                    continue;
                };
                printf("\n Done: %d\n", i);
                time = time_temp;
            }
        }
#endif
#pragma omp section
        {
            omp_set_num_threads(M);

#ifndef VER_X
            for (i = 0; i < 100; i++) /* 100 экспериментов */
#else
            for (i = 0; i < 5; i++)
#endif
            {

                SET_SEED_POINT(pSeed, i);

                for (j = 0; j < N; j++) // [1; A]
                    M1[j] = RANDOM_DOUBLE(1, A, &pSeed);

                for (j = 0; j < N2; j++) // [A; 10*A]
                    M2[j] = RANDOM_DOUBLE(A, 10 * A, &pSeed);

                X = 0;
#pragma omp parallel default(none) private(j) shared(N, N2, M1, M2, M2Copy, minimum) reduction(+ : X)
                {
#pragma omp for
                    for (j = 0; j < N; j++) // 2) Map. 6: Кубический корень после деления на число e
                        M1[j] = cbrt(M1[j] / M_E);

#pragma omp for
                    for (j = 0; j < N2; j++) // Копия массива M2
                        M2Copy[j + 1] = M2[j];

#pragma omp for
                    for (j = 0; j < N2; j++) // 2: Элементы нового массива равны модулю косинуса от суммы текущего и прошлого элементов массива
                        M2[j] = fabs(cos(M2Copy[j + 1] + M2Copy[j]));

#pragma omp for
                    for (j = 0; j < N2; j++) // 3) Merge. 4: Выбор большего
                        M2[j] = MAX(M1[j], M2[j]);

#pragma omp single
                    {
                        sort(M2, N2, M2Copy); // 4) Sort. 6: Сортировка вставками
                        minimum = M2[0];
                    }

#pragma omp for
                    for (j = 0; j < N2; j++) // 5) Reduce. Сумма синусов элементов, которые при делении на минимум дают чётное число в целой части
                        if (IS_WHOLE_PART_EVEN(M2Copy[j] / minimum))
                            X += sin(M2Copy[j]);
#pragma omp barrier
                }
#ifdef VER_X
                printf("X = %f\n", X);
#endif
            }
            finished = 1;
        }
    }
    T2 = omp_get_wtime();
    printf("\n%f\n", (T2 - T1) * 1000.0);

    finished = 1;

    free(M2Copy);
    free(M2);
    free(M1);

    return 0;
}