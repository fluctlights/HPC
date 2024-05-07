import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate execution time chart from Excel data.')
    parser.add_argument('--directories', nargs='+', type=str, help='Directories to save the output files')
    parser.add_argument('--OpenMP', type=str, default='False', help='Whether OpenMP was used (True or False)')
    args = parser.parse_args()

    for directory in args.directories:
        file_path = os.path.join(directory, 'results.xlsx')
        df = pd.read_csv(file_path, sep=";")

        if args.OpenMP.lower() == 'true':
            df_selected = df[['threads', 'k', 'average_time']]
        else:
            df_selected = df[['k', 'average_time']]

        df_sorted = df_selected.sort_values(by='k').reset_index(drop=True)


        if args.OpenMP.lower() == 'true':
            x_labels = [f"{clusters} & {threads}" for clusters, threads in zip(df_sorted['k'], df_sorted['threads'])]
            plt.figure(figsize=(10, 6))
            plt.bar(df_sorted.index, df_sorted['average_time'], color='skyblue')
            plt.title('Execution Time')
            plt.xlabel('Number of Clusters (K) && threads')
            plt.ylabel('Execution Time (seconds)')
            plt.xticks(df_sorted.index, x_labels, rotation=45, ha='right')
            plt.grid(axis='y')
        else:
            plt.figure(figsize=(8, 6))
            plt.bar(df_sorted['K'], df_sorted['average_time'], color='skyblue')
            plt.title('Execution Time')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Execution Time (seconds)')
            plt.grid(axis='y')

        output_path = os.path.join(directory, 'kmeans_chart_results.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')