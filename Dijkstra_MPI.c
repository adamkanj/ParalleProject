#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>
#define N 100
#define INF INT_MAX
int graph[N][N], dist[N], visited[N];
int get_min_distance_vertex(int rank) {
    int i, min = INF, min_index;
    for (i = 0; i < N; i++) {
        if (!visited[i] && dist[i] <= min) {
            min = dist[i];
            min_index = i;
        }
    }return min_index;
}
void init() {
    int i;
    for (i = 0; i < N; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
}
void dijkstra(int rank, int num_procs) {
    int i, j, u, start, end, num_vertices = 0;
    if (rank == 0) {
        start = 0;
        end = N / num_procs;
    } else if (rank == num_procs - 1) {
        start = rank * (N / num_procs);
        end = N;
    } else {
        start = rank * (N / num_procs);
        end = start + (N / num_procs);
    }
    for (i = start; i < end; i++) {
        dist[i] = graph[0][i];
    }
    visited[0] = 1;
    for (i = 0; i < N - 1; i++) {
        u = get_min_distance_vertex(rank);
        if (u >= start && u < end) {
            visited[u] = 1;
            num_vertices++;}
        MPI_Bcast(&visited, N, MPI_INT, 0, MPI_COMM_WORLD);
        for (j = start; j < end; j++) {
            if (!visited[j] && graph[u][j] && dist[u] != INF
                    && dist[u] + graph[u][j] < dist[j]) {
                dist[j] = dist[u] + graph[u][j];
            }
        }
    }int total_vertices;
    MPI_Reduce(&num_vertices, &total_vertices, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Number of vertices: %d\n", N);
    }
}
int main(int argc, char *argv[]) {
    int rank, num_procs, i, j;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                graph[i][j] = 0;
            } else {
                graph[i][j] = rand() % 10;
            }
        }
    }init();
    double start_time = MPI_Wtime();
    dijkstra(rank, num_procs);
    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Runtime: %f ms\n", (end_time - start_time) * 1000);
    }MPI_Finalize();
    return 0;
}
