import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate execution time chart from Excel data.')
    parser.add_argument('--directories', nargs='+', type=str, help='Directories to save the output files')
    parser.add_argument('--mpi', type=str, default='False', help='Whether MPI was used (True or False)')
    args = parser.parse_args()

    for directory in args.directories:
        file_path = os.path.join(directory, 'kmeans_results.xlsx')
        df = pd.read_excel(file_path)

        if args.mpi.lower() == 'true':
            df_selected = df[['Processors', 'K', 'Execution Time (seconds)']]
        else:
            df_selected = df[['K', 'Execution Time (seconds)']]

        df_sorted = df_selected.sort_values(by='K')

        if args.mpi.lower() == 'true':
            x_labels = [f"{clusters} & {processor}" for clusters, processor in zip(df_sorted['K'], df_sorted['Processors'])]
            plt.figure(figsize=(10, 6))
            plt.bar(df_sorted.index, df_sorted['Execution Time (seconds)'], color='skyblue')
            plt.title('Execution Time')
            plt.xlabel('Number of Clusters (K) && Processors')
            plt.ylabel('Execution Time (seconds)')
            plt.xticks(df_sorted.index, x_labels, rotation=45, ha='right')
            plt.grid(axis='y')
        else:
            plt.figure(figsize=(8, 6))
            plt.bar(df_sorted['K'], df_sorted['Execution Time (seconds)'], color='skyblue')
            plt.title('Execution Time')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Execution Time (seconds)')
            plt.grid(axis='y')

        output_path = os.path.join(directory, 'kmeans_chart_results.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')