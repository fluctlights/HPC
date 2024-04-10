import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys
import time
from mpi4py import MPI

DIRECTORY = "kmeans/mpi/macqueenV1"

class MacQueen:
    def __init__(self):
        if rank == 0:
            print("Loading data...")
            sys.stdout.flush()
            self.data = np.loadtxt('pavia.txt', delimiter=',')
            self.data_array = np.array(self.data)
            self.matrix = np.reshape(self.data_array, (783640, 102))
            self.pixels = self.matrix.astype(np.float32)
            print("Data loaded.")
            sys.stdout.flush()


    def _initialize_centroids(self, X, k):
        """Randomly initialize centroids."""
        n_samples, _ = X.shape
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[indices]
        return centroids


    def _calculate_distance(self, x1, x2, distance_metric='euclidean'):
        """Calculate distance between two points."""
        if distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif distance_metric == 'chebyshev':
            return np.max(np.abs(x1 - x2))
        else:
            raise ValueError("Invalid distance metric. Please choose 'euclidean', 'manhattan', or 'chebyshev'.")


    def _assign_clusters(self, X, centroids, distance_metric='euclidean'):
        """Assign each data point to the closest centroid based on chosen distance metric."""
        distances = np.zeros((X.shape[0], len(centroids)))
        for i, centroid in enumerate(centroids):
            for j, x in enumerate(X):
                distances[j, i] = self._calculate_distance(x, centroid, distance_metric)
        return np.argmin(distances, axis=1)


    def _update_centroids(self, X, clusters, k):
        """Update centroids based on the mean of data points assigned to each cluster."""
        centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            centroids[i] = np.mean(X[clusters == i], axis=0)
        return centroids


    def kmeans(self, k, distance_metric='euclidean', max_iters=100, centroid_tolerance=1e-8, image=False):
        """MacQueen's K-Means Algorithm."""
        if rank == 0:
            centroids = self._initialize_centroids(self.pixels, k)

            chunk_size, remainder = divmod(len(self.pixels), size)
            pixel_chunks = [self.pixels[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(size)]
        else:
            centroids = None
            pixel_chunks = None

        centroids = comm.bcast(centroids, root=0)
        recvbuf_pixels_chunks = comm.scatter(pixel_chunks, root=0)

        for _ in range(max_iters):
            clusters = self._assign_clusters(recvbuf_pixels_chunks, centroids, distance_metric)

            new__local_centroids = self._update_centroids(recvbuf_pixels_chunks, clusters, k)
            all_local_centroids = comm.gather(new__local_centroids, root=0)

            if rank == 0:
                new_centroids = np.mean(all_local_centroids, axis=0)
                centroids_diff = np.linalg.norm(new_centroids - centroids)
            else:
                centroids_diff = None

            centroids_diff = comm.bcast(centroids_diff, root=0)

            if centroids_diff < centroid_tolerance:
                break

            if rank == 0:
                centroids = new_centroids

            centroids = comm.bcast(centroids, root=0)

        all_clusters = comm.gather(clusters, root=0)

        if rank == 0:
            all_clusters_merged = np.concatenate(all_clusters, axis=0)

            if image is True:
                generate_image(all_clusters_merged)

            return all_clusters_merged, centroids


    def timed_kmeans(self, k, distance_metric='euclidean', max_iters=100, centroid_tolerance=1e-8, image=False, repetitions=1):
        """Wrapper function to run kmeans and measure execution time."""

        if rank == 0:
            total_execution_times = []

        for repetition in range(repetitions):
            if rank == 0:
                print("Executing K-Means", repetition + 1, "...")
                sys.stdout.flush()
                start_time = time.time()
                
            self.kmeans(k, distance_metric, max_iters, centroid_tolerance, False)

            if rank == 0:
                end_time = time.time()
                execution_time = end_time - start_time
                total_execution_times.append(execution_time)
                print("Total execution time:", execution_time, "seconds")
                sys.stdout.flush()

        if rank == 0:
            print("Average execution time:", np.mean(total_execution_times), "seconds")
            sys.stdout.flush()

            df = pd.DataFrame({
                "Processors": [size],
                "K": [k],
                "Distance Metric": [distance_metric],
                "Max Iters": [max_iters],
                "Centroid Tolerance": [centroid_tolerance],
                "Repetitions": [repetitions],
                "Execution Time (seconds)": [np.mean(total_execution_times)]
            })

            if not os.path.exists(DIRECTORY):
                os.makedirs(DIRECTORY)

            output_file = os.path.join(DIRECTORY, "kmeans_results.xlsx")
            if os.path.isfile(output_file):
                existing_df = pd.read_excel(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_excel(output_file, index=False)
   

def generate_image(labels):
    cluster_map = np.reshape(labels, (715,1096)).astype(np.uint8)
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
        (255, 255, 255), # White
        (0, 0, 0),     # Black
        (128, 128, 128), # Gray
        (64, 64, 64)   # Gray
    ]

    color_map_array = np.array(colors)
    result_image = np.zeros((1096,715, 3), dtype=np.uint8)

    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            result_image[i, j] = color_map_array[cluster_map[j, i]]

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    cv2.imwrite(os.path.join(DIRECTORY, "paviakmeanspy.jpg"), result_image)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="K-Means clustering with MPI support")
    parser.add_argument('--k', type=int, help='Number of clusters')
    parser.add_argument('--distance_metric', type=str, default='euclidean', help='Distance metric to use')
    parser.add_argument('--max_iters', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--centroid_tolerance', type=float, default=1e-8, help='Tolerance for centroid convergence')
    parser.add_argument('--image', type=bool, help='Generate image of clustered data')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions')

    args = parser.parse_args()

    macQueen = MacQueen()
    macQueen.timed_kmeans(args.k, args.distance_metric, args.max_iters, args.centroid_tolerance, args.image, args.repetitions)