#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace std::chrono;
using namespace cv;

const int EUCLIDEAN = 0;
const int MANHATTAN = 1;
const int CHEBYSHEV = 2;

const string DIRECTORY = "kmeans/sequential/";

void readDataFromFile(const string& filename, Mat& matrix) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return;
    }

    string line;
    unsigned int row = 0, col = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        string element;
        while (getline(iss, element, ',')) {
            float value;
            stringstream(element) >> value;
            matrix.at<float>(row, col) = value;
            ++col;
            if (col >= matrix.size().width) {
                col = 0;
                ++row;
            }
            if (row >= matrix.size().height)
                break;
        }
    }
    file.close();
}

void initialize_centroids(const Mat& data, int k, Mat& centroids) {
    vector<int> indices(k);
    for (int i = 0; i < k; ++i) {
        indices[i] = rand() % data.rows;
        data.row(indices[i]).copyTo(centroids.row(i));
    }
}

double calculate_distance(const float* point1, const float* point2, int n_features, int distance_metric) {
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

void assign_clusters(const Mat& data, int k, const Mat& centroids, Mat& labels, int distance_metric) {
    for (int i = 0; i < data.rows; ++i) {
        double min_distance = INFINITY;
        for (int j = 0; j < k; ++j) {
            double distance = calculate_distance(data.ptr<float>(i), centroids.ptr<float>(j), data.cols, distance_metric);
            if (distance < min_distance) {
                min_distance = distance;
                labels.at<int>(i, 0) = j;
            }
        }
    }
}

void update_centroids(const Mat& data, const vector<int>& clusters, int k, Mat& centroids) {
    vector<int> cluster_sizes(k, 0);
    centroids.setTo(0);
    for (int i = 0; i < data.rows; ++i) {
        int cluster_index = clusters[i];
        cluster_sizes[cluster_index]++;
        centroids.row(cluster_index) += data.row(i);
    }
    for (int i = 0; i < k; ++i) {
        if (cluster_sizes[i] != 0) {
            centroids.row(i) /= cluster_sizes[i];
        }
    }
}

void kmeans(const Mat& data, int k, int distance_metric, int max_iters, double centroid_tolerance, Mat& labels, Mat& centroids) {
    initialize_centroids(data, k, centroids);
    bool converged = false;
    for (int iter = 0; iter < max_iters && !converged; ++iter) {
        Mat prev_centroids;
        centroids.copyTo(prev_centroids);
        assign_clusters(data, k, centroids, labels, distance_metric);
        update_centroids(data, labels, k, centroids);
        double max_diff = 0;
        for (int i = 0; i < k; ++i) {
            double diff = norm(centroids.row(i), prev_centroids.row(i));
            max_diff = max(max_diff, diff);
        }
        if (max_diff <= centroid_tolerance)
            converged = true;
    }
}

void generateImage(Mat& labels, Mat &centroids) {

    labels = labels.reshape(1,715);

    Mat cluster_map(715,1096, CV_8U);

    int index = 0;
    int prev_label = 0;

    for (int i = 0; i < cluster_map.rows; ++i) {
      for (int j = 0; j < cluster_map.cols; ++j) {
            int l = cluster_map.at<uchar>(i, j) = labels.at<int>(i,j);
            if (prev_label != cluster_map.at<unsigned char>(i, j)){
                prev_label = cluster_map.at<uchar>(i, j);
            }
        }
    }

    Mat result_image(715,1096, CV_8UC3);

    //  Mapa de color (B, G, R) en OpenCV
    const Vec<uchar, 3> colors[] = {
        Vec<uchar, 3>(0, 0, 255),   // Red
        Vec<uchar, 3>(0, 255, 0),   // Green
        Vec<uchar, 3>(255, 0, 0),   // Blue
        Vec<uchar, 3>(255, 255, 0), // Yellow
        Vec<uchar, 3>(255, 0, 255), // Magenta
        Vec<uchar, 3>(0, 255, 255), // Cyan
        Vec<uchar, 3>(255, 255, 255), // White
        Vec<uchar, 3>(0, 0, 0),     // Black
        Vec<uchar, 3>(128, 128, 128), // Gray
	    Vec<uchar, 3>(64, 64, 64) // Gray
    };

    index = 0;
    for (int i = 0; i < result_image.rows; ++i) {
      for (int j = 0; j < result_image.cols; ++j) {
          uchar cluster_index = cluster_map.at<unsigned char>(index++);
          result_image.at<Vec3b>(i, j) = colors[cluster_index];
      }
    }

    Mat resultT;
    transpose(result_image,resultT);

    if (!filesystem::exists(DIRECTORY)) {
        filesystem::create_directories(DIRECTORY);
    }

    string file_path = DIRECTORY + "paviakmeanscpp.jpg";

    imwrite(file_path,resultT);
    waitKey(0);
}

void timed_kmeans(int k, int distance_metric, int max_iters, double centroid_tolerance, int repetitions, int image) {
    string filename = "pavia.txt";
    Mat data(783640, 102, CV_32F);

    readDataFromFile(filename, data);

    Mat labels(data.rows, 1, CV_32S);
    Mat centroids(k, data.cols, CV_32F);

    cout << "Starting K-means" << endl;

    duration<double> total_time = duration<double>::zero();

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        cout << "Executing K-Means repetition " << repetition + 1 << "..." << endl;
        auto start_time = high_resolution_clock::now();
        kmeans(data, k, distance_metric, max_iters, centroid_tolerance, labels, centroids);
        auto end_time = high_resolution_clock::now();
        duration<double> elapsed_time = end_time - start_time;
        cout << "Execution time for repetition " << repetition + 1 << ": " << elapsed_time.count() << " seconds" << endl;
        total_time += elapsed_time;
        if(image){
            generateImage(labels, centroids);
        }
    }

    double average_time = total_time.count() / repetitions;
    cout << "Average execution time: " << average_time << " seconds" << endl;
    
    if (!filesystem::exists(DIRECTORY)) {
        filesystem::create_directories(DIRECTORY);
    }

    string file_path = DIRECTORY + "results.xlsx";

    ofstream file(file_path, ios::app);
    bool file_exists = file.tellp() != 0;
    if (!file_exists) {
        file << "k;distance_metric;max_iters;centroid_tolerance;repetitions;average_time" << endl;
    }
    file << k << ";" << distance_metric << ";" << max_iters << ";" << centroid_tolerance << ";" << repetitions << ";" << average_time << endl;
    file.close();
}

int main(int argc, char *argv[]) {
    int k = 2;
    int distance_metric = EUCLIDEAN;
    int max_iters = 100;
    double centroid_tolerance = 0.2;
    int repetitions = 1;
    int image = 1;

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
                        if (argc > 6) {
                            image = atoi(argv[6]);
                        }
                    }
                }
            }
        }
    }

    timed_kmeans(k, distance_metric, max_iters, centroid_tolerance, repetitions, image);

    return 0;
}