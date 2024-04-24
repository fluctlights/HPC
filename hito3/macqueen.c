#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define EUCLIDEAN 0
#define MANHATTAN 1
#define CHEBYSHEV 2

typedef struct {
    float** data;
    int rows;
    int cols;
} Mat;

float** allocate_2d_array(int rows, int cols) {
    float** arr = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; ++i) {
        arr[i] = (float*)malloc(cols * sizeof(float));
    }
    return arr;
}

void free_2d_array(float** arr, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(arr[i]);
    }
    free(arr);
}

void readDataFromFile(const char* filename, Mat* matrix) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return;
    }

    for (int row = 0; row < matrix->rows; ++row) {
        for (int col = 0; col < matrix->cols; ++col) {
            float value;
            if (fscanf(file, "%f,", &value) != 1) {
                fprintf(stderr, "Error: Could not read value at row %d, column %d\n", row, col);
                fclose(file);
                return;
            }
            matrix->data[row][col] = value;
        }
    }

    fclose(file);
}

void initialize_centroids(float** X, int n_samples, int n_features, int k, float** centroids) {
    int* indices = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; ++i) {
        indices[i] = rand() % n_samples;
        for (int j = 0; j < n_features; ++j) {
            centroids[i][j] = X[indices[i]][j];
        }
    }
    free(indices);
}

double calculate_distance(float* point1, float* point2, int n_features, int distance_metric) {
    double distance = 0.0;
    if (distance_metric == EUCLIDEAN) {
        for (int i = 0; i < n_features; ++i) {
            distance += pow(point1[i] - point2[i], 2);
        }
        distance = sqrt(distance);
    } else if (distance_metric == MANHATTAN) {
        for (int i = 0; i < n_features; ++i) {
            distance += fabs(point1[i] - point2[i]);
        }
    } else if (distance_metric == CHEBYSHEV) {
        distance = fabs(point1[0] - point2[0]);
        for (int i = 1; i < n_features; ++i) {
            double temp = fabs(point1[i] - point2[i]);
            if (temp > distance)
                distance = temp;
        }
    }
    return distance;
}

void assign_clusters(float** X, int n_samples, int n_features, float** centroids, int k, int* clusters, int distance_metric) {
    for (int i = 0; i < n_samples; ++i) {
        double min_distance = INFINITY;
        for (int j = 0; j < k; ++j) {
            double distance = calculate_distance(X[i], centroids[j], n_features, distance_metric);
            if (distance < min_distance) {
                min_distance = distance;
                clusters[i] = j;
            }
        }
    }
}

void update_centroids(float** X, int n_samples, int n_features, int* clusters, int k, float** centroids) {
    for (int i = 0; i < k; ++i) {
        int count = 0;
        double* centroid_sum = (double*)calloc(n_features, sizeof(double));
        for (int j = 0; j < n_samples; ++j) {
            if (clusters[j] == i) {
                for (int f = 0; f < n_features; ++f) {
                    centroid_sum[f] += X[j][f];
                }
                count++;
            }
        }
        for (int f = 0; f < n_features; ++f) {
            centroids[i][f] = centroid_sum[f] / count;
        }
        free(centroid_sum);
    }
}

void kmeans(float** X, int n_samples, int n_features, int k, int distance_metric, int max_iters, double centroid_tolerance, int* clusters, float** centroids, double* time_taken) {
    clock_t start_time = clock();
    initialize_centroids(X, n_samples, n_features, k, centroids);
    bool converged = false;
    for (int iter = 0; iter < max_iters && !converged; ++iter) {
        float** prev_centroids = allocate_2d_array(k, n_features);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n_features; ++j) {
                prev_centroids[i][j] = centroids[i][j];
            }
        }
        assign_clusters(X, n_samples, n_features, centroids, k, clusters, distance_metric);
        update_centroids(X, n_samples, n_features, clusters, k, centroids);
        converged = true;
        for (int i = 0; i < k && converged; ++i) {
            for (int j = 0; j < n_features; ++j) {
                if (fabs(prev_centroids[i][j] - centroids[i][j]) > centroid_tolerance) {
                    converged = false;
                    break;
                }
            }
        }
        free_2d_array(prev_centroids, k);
    }
    clock_t end_time = clock();
    *time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
}

void timed_kmeans(int k, int distance_metric, int max_iters, double centroid_tolerance, int repetitions) {
    const char* filename = "pavia.txt";
    Mat data;
    data.rows = 783640;
    data.cols = 102;

    data.data = (float**)malloc(data.rows * sizeof(float*));
    for (int i = 0; i < data.rows; ++i) {
        data.data[i] = (float*)malloc(data.cols * sizeof(float));
    }
    readDataFromFile(filename, &data);

    int n_samples = data.rows;
    int n_features = data.cols;

    int* clusters = (int*)malloc(n_samples * sizeof(int));
    float** centroids = allocate_2d_array(k, n_features);
    double time_taken;

    double total_execution_time = 0.0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        printf("Executing K-Means repetition %d...\n", repetition + 1);
        kmeans(data.data, n_samples, n_features, k, distance_metric, max_iters, centroid_tolerance, clusters, centroids, &time_taken);
        total_execution_time += time_taken;
        printf("Repetition %d execution time: %.2f seconds\n", repetition + 1, time_taken);
    }

    for (int i = 0; i < k; ++i) {
        printf("Centroid %d: ", i + 1);
        for (int j = 0; j < n_features; ++j) {
            printf("%.2f ", centroids[i][j]);
        }
        printf("\n");
    }

    printf("Labels: ");
    for (int i = 0; i < n_samples; ++i) {
        printf("%d ", clusters[i]);
    }
    printf("\n");

    printf("Average execution time over %d repetitions: %.2f seconds\n", repetitions, total_execution_time / repetitions);

    FILE* csv_file = fopen("kmeans_results.csv", "a");
    if (csv_file) {
        if (ftell(csv_file) == 0) {
            fprintf(csv_file, "K;Distance Metric;Max Iters;Centroid Tolerance;Repetitions;Execution Time (seconds)\n");
        }
        fprintf(csv_file, "%d;%d;%d;%f;%d;%.2f\n", k, distance_metric, max_iters, centroid_tolerance, repetitions, total_execution_time / repetitions);
        fclose(csv_file);
    } else {
        fprintf(stderr, "Error: Could not create CSV file\n");
    }

    free_2d_array(data.data, data.rows);
    free(clusters);
    free_2d_array(centroids, k);
}

int main(int argc, char *argv[]) {
    int k = 3;
    int distance_metric = EUCLIDEAN;
    int max_iters = 100;
    double centroid_tolerance = 1e-8;
    int repetitions = 1;

    if (argc > 1) {
        k = atoi(argv[1]);
        if (argc > 2) {
            distance_metric = atoi(argv[2]);
            if (argc > 3) {
                max_iters = atoi(argv[3]);
                if (argc > 4) {
                    centroid_tolerance = atof(argv[4]);
                    if (argc > 5) {
                        repetitions = atoi(argv[5]);
                    }
                }
            }
        }
    }

    timed_kmeans(k, distance_metric, max_iters, centroid_tolerance, repetitions);

    return 0;
}