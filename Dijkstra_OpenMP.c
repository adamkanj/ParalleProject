#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <limits.h> 
#define n_node 3000
int matrix_distance[n_node][n_node];
int final_matrix_distance[n_node][n_node];
int main(int argc, char** argv[]) {
  int n_thread, use_serial, source, itr;
  n_thread = atoi(argv[1]);
  use_serial = atoi(argv[2]);
  int seed = 29;
  clock_t t_serial;
  double t_start_parallel, t_end_parallel;
  init_graph(seed);
  if (use_serial == 1) { // START SERIAL DIJKSTRA ALGORITHM
    t_serial = clock();
    for (itr = 0; itr < n_node; itr++) {  
      dijkstra(itr, final_matrix_distance[itr]);
      printf("Serial | Node %d out of %d\n", itr, n_node);
    }
    t_serial = clock() - t_serial;
    double time_taken_serial = ((double)t_serial * 1000000) / (CLOCKS_PER_SEC); // PRINT RESULT OF SERIAL DIJKSTRA ALGORITHM
    printf("\n%s%2.f%s\n", "Time elapsed for serial dijkstra algorithm: ", time_taken_serial, " microsecond"); // END OF SERIAL DIJKSTRA ALGORITHM
  } else if (use_serial == 0) { // START PARALLEL DIJKSTRA ALGORITHM USING OPENMP
    t_start_parallel = omp_get_wtime();
    #pragma omp parallel num_threads(n_thread)
    openmp_dijkstra();
    t_end_parallel = omp_get_wtime();
    double time_taken_openmp = (t_end_parallel - t_start_parallel) * 1000000;
    printf("\n%s%2.f%s\n", "Time elapsed for OpenMP parallel dijkstra algorithm: ", time_taken_openmp, " microsecond");
    print_matrix_to_file();
  }
  return 0;
}
void openmp_dijkstra() {
  int rank = omp_get_thread_num();
  int n_thread = omp_get_num_threads();
  int itr; 
  for (itr = rank; itr < n_node; itr += n_thread) {
    dijkstra(itr, final_matrix_distance[itr]);
    printf("Thread %d | Node %d out of %d\n", rank, itr, n_node);
  }
}
void init_graph(int seed) {
  for (int i = 0; i < n_node; i++) {
    for (int j = 0; j < n_node; j++) {
      if (i == j) {
        matrix_distance[i][j] = 0;
      } else if (i < j) {
        int parity = rand() % seed;
        if (parity % 2 == 0) {
          matrix_distance[i][j] = -1;
          matrix_distance[j][i] = -1;
        } else {
          matrix_distance[i][j] = parity;
          matrix_distance[j][i] = parity;
        }
      }
    }
  }
}
void print_matrix_to_file() {
  FILE * fp;
  fp = fopen ("../output/result_3000.txt","w");
  for (int i = 0; i < n_node; i++) {
    for (int j = 0; j < n_node; j++) {
      fprintf(fp, "%d ", final_matrix_distance[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose (fp);
}
int minDistance(int dist[], bool sptSet[]) { 
  int min = INT_MAX, min_index; 
  for (int v = 0; v < n_node; v++) {
    if (sptSet[v] == false && dist[v] <= min) {
      min = dist[v], min_index = v; 
    }
  }
  return min_index; 
} 
void dijkstra(int src, int dist[n_node]) { 
  bool sptSet[n_node];
  for (int i = 0; i < n_node; i++) {
    dist[i] = INT_MAX, sptSet[i] = false; 
  }
  dist[src] = 0; 
  for (int count = 0; count < n_node - 1; count++) { 
    int u = minDistance(dist, sptSet); 
    sptSet[u] = true; 
    for (int v = 0; v < n_node; v++) {
      if (!sptSet[v] && matrix_distance[u][v] && dist[u] != INT_MAX 
        && dist[u] + matrix_distance[u][v] < dist[v]) {
        dist[v] = dist[u] + matrix_distance[u][v]; 
      }
    }
  } 
}
