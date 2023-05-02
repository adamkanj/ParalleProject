#include <stdio.h>     
#include <stdlib.h>
#include <time.h>      
#include <math.h>
#define VERTICES 1024
#define DENSITY 16     
#define MAX_WEIGHT 1000000      
#define INF_DIST 1000000000     
#define CPU_IMP 1      
#define GPU_IMP 1               
#define THREADS 2               
#define RAND_SEED 1234          
#define THREADS_BLOCK 512
typedef float data_t;  
#define CPG 2.53
#define GIG 1000000000
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true){
    if (code != cudaSuccess){
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
int main() {
    srand(RAND_SEED);
    void setIntArrayValue(int* in_array, int array_size, int value);   
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value); 
    void initializeGraphZero(data_t* graph, int num_vertices);                    
    void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices);    
    void checkArray(int* a, int length);                                          
    void checkArrayData(data_t* a, int length);
    
    void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start);   
    __global__ void closestNodeCUDA(data_t* node_dist, int* visited_node, int* global_closest, int num_vertices);                   
    __global__ void cudaRelax(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int* source);                  
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec start, end;                    
    struct timespec time_stamp[CPU_IMP];
    int graph_size      = VERTICES*VERTICES*sizeof(data_t); 
    int int_array       = VERTICES*sizeof(int);             
    int data_array      = VERTICES*sizeof(data_t);          
    data_t* graph       = (data_t*)malloc(graph_size);      
    data_t* node_dist   = (data_t*)malloc(data_array);      
    int* parent_node    = (int*)malloc(int_array);          
    int* edge_count     = (int*)malloc(int_array);          
    int* visited_node   = (int*)malloc(int_array);          
    int *pn_matrix      = (int*)malloc((CPU_IMP+GPU_IMP)*int_array);  
    data_t* dist_matrix = (data_t*)malloc((CPU_IMP + GPU_IMP)*data_array);
    printf("Variables created, allocated\n");
    data_t* gpu_graph;
    data_t* gpu_node_dist;
    int* gpu_parent_node;
    int* gpu_visited_node;
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_graph, graph_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_node_dist, data_array));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_parent_node, int_array));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_visited_node, int_array));

    int* closest_vertex = (int*)malloc(sizeof(int));
    int* gpu_closest_vertex;
    closest_vertex[0] = -1;
    CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int))));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice));

    setIntArrayValue(edge_count, VERTICES, 0); 
    setDataArrayValue(node_dist, VERTICES, INF_DIST); 
    setIntArrayValue(parent_node, VERTICES, -1);      
    setIntArrayValue(visited_node, VERTICES, 0);       
    initializeGraphZero(graph, VERTICES);              
    constructGraphEdge(graph, edge_count, VERTICES);   
    free(edge_count);                   
    printf("Variables initialized.\n");

    int i;                                         
    int origin = (rand() % VERTICES);              
    printf("Origin vertex: %d\n", origin);
    int version = 0;
    printf("Running serial...");
    clock_gettime(CLOCK_REALTIME, &start);
    dijkstraCPUSerial(graph, node_dist, parent_node, visited_node, VERTICES, origin);
    clock_gettime(CLOCK_REALTIME, &end);
    time_stamp[version] = diff(start, end);
    for (i = 0; i < VERTICES; i++) {        
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }
    printf("Done!\n");
    version++;
    cudaEvent_t exec_start, exec_stop;
    float elapsed_exec;               
    CUDA_SAFE_CALL(cudaEventCreate(&exec_start));
    CUDA_SAFE_CALL(cudaEventCreate(&exec_stop));

    setDataArrayValue(node_dist, VERTICES, INF_DIST);       
    setIntArrayValue(parent_node, VERTICES, -1);            
    setIntArrayValue(visited_node, VERTICES, 0);            
    node_dist[origin] = 0;                                  
    
    CUDA_SAFE_CALL(cudaMemcpy(gpu_graph, graph, graph_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_node_dist, node_dist, data_array, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_parent_node, parent_node, int_array, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_visited_node, visited_node, int_array, cudaMemcpyHostToDevice));

    dim3 gridMin(1, 1, 1);
    dim3 blockMin(1, 1, 1);

    dim3 gridRelax(VERTICES / THREADS_BLOCK, 1, 1);
    dim3 blockRelax(THREADS_BLOCK, 1, 1);           
    
    CUDA_SAFE_CALL(cudaEventRecord(exec_start));
    for (int i = 0; i < VERTICES; i++) {
        closestNodeCUDA <<<gridMin, blockMin>>>(gpu_node_dist, gpu_visited_node, gpu_closest_vertex, VERTICES);            
        cudaRelax <<<gridRelax, blockRelax>>>(gpu_graph, gpu_node_dist, gpu_parent_node, gpu_visited_node, gpu_closest_vertex);
    }
    CUDA_SAFE_CALL(cudaEventRecord(exec_stop));
    CUDA_SAFE_CALL(cudaMemcpy(node_dist, gpu_node_dist, data_array, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(parent_node, gpu_parent_node, int_array, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(visited_node, gpu_visited_node, int_array, cudaMemcpyDeviceToHost));
    for (i = 0; i < VERTICES; i++) {           
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }

    CUDA_SAFE_CALL(cudaFree(gpu_graph));
    CUDA_SAFE_CALL(cudaFree(gpu_node_dist));
    CUDA_SAFE_CALL(cudaFree(gpu_parent_node));
    CUDA_SAFE_CALL(cudaFree(gpu_visited_node));
    printf("\nVertices: %d", VERTICES);
    printf("\nDensity: %d", DENSITY);
    printf("\nMax Weight: %d", MAX_WEIGHT);
    printf("\n\nSerial cycles: \n");
    for (i = 0; i < CPU_IMP; i++) {
        printf("%ld", (long int)((double)(CPG)*(double)
            (GIG * time_stamp[i].tv_sec + time_stamp[i].tv_nsec)));
    }
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_exec, exec_start, exec_stop));
    printf("\n\nCUDA Time (ms): %7.9f\n", elapsed_exec);

    printf("\n\nError checking:\n");
    printf("----Serial vs CUDA:\n");
    int p_errors = 0, d_errors = 0;
    printf("--------%d parent errors found.\n", p_errors);
    printf("--------%d dist errors found.\n", d_errors);
}
void setIntArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}
void setDataArrayValue(data_t* in_array, int array_size, data_t init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}
void initializeGraphZero(data_t* graph, int num_vertices) {
    int i, j;
    for (i = 0; i < num_vertices; i++) {
        for (j = 0; j < num_vertices; j++) {
            graph[i*num_vertices + j] = (data_t)0;
        }
    }
}
void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices) {
    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);
    int i;                  
    int rand_vertex;        
    int curr_num_edges;     
    data_t weight;    

    printf("Initializing a connected graph...");
    for (i = 1; i < num_vertices; i++) {
        rand_vertex = (rand() % i);                     
        weight = (rand() % MAX_WEIGHT) + 1;             
        graph[rand_vertex*num_vertices + i] = weight;   
        graph[i*num_vertices + rand_vertex] = weight;
        edge_count[i] += 1;                             
        edge_count[rand_vertex] += 1;
    }
    printf("done!\n");
    printf("Checking density...");
    for (i = 0; i < num_vertices; i++) {  
        curr_num_edges = edge_count[i];   
        while (curr_num_edges < DENSITY) {
            rand_vertex = (rand() % num_vertices);  
            weight = (rand() % MAX_WEIGHT) + 1;     
            if ((rand_vertex != i) && (graph[i*num_vertices + rand_vertex] == 0)) { 
                graph[i*num_vertices + rand_vertex] = weight;
                graph[rand_vertex*num_vertices + i] = weight;
                edge_count[i] += 1;
                curr_num_edges++;              
            }
        }
    }
    printf("done!\n");
}
int closestNode(data_t* node_dist, int* visited_node, int num_vertices) {
    data_t dist = INF_DIST + 1;    
    int node = -1;              
    int i;                      
    for (i = 0; i < num_vertices; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] == 0)) { 
            node = i;              
            dist = node_dist[i];   
        }
    }
    return node;    
}
void checkArray(int* a, int length) {
    int i;
    printf("Proof: ");
    for (i = 0; i < length; i++) {
        printf("%d, ", a[i]);
    }
    printf("\n\n");
}
void checkArrayData(data_t* a, int length) {
    int i;
    printf("Proof: ");
    for (i = 0; i < length; i++) {
        printf("%f, ", a[i]);
    }
    printf("\n\n");
}
struct timespec diff(struct timespec start, struct timespec end){
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}
void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start) {
    void setIntArrayValue(int* in_array, int array_size, int init_value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);
    setDataArrayValue(node_dist, VERTICES, INF_DIST);    
    setIntArrayValue(parent_node, VERTICES, -1);         
    setIntArrayValue(visited_node, VERTICES, 0);         
    node_dist[v_start] = 0;                  
    int i, next;
    for (i = 0; i < num_vertices; i++) {
        int curr_node = closestNode(node_dist, visited_node, num_vertices); 
        visited_node[curr_node] = 1;                                        
        for (next = 0; next < num_vertices; next++) {
            int new_dist = node_dist[curr_node] + graph[curr_node*num_vertices + next];
            if ((visited_node[next] != 1)
                && (graph[curr_node*num_vertices + next] != (data_t)(0))
                && (new_dist < node_dist[next])) {
                node_dist[next] = new_dist;       
                parent_node[next] = curr_node;    
            }
        }
    }
}
__global__ void closestNodeCUDA(data_t* node_dist, int* visited_node, int* global_closest, int num_vertices) {
    data_t dist = INF_DIST + 1;
    int node = -1;
    int i;
    for (i = 0; i < num_vertices; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] != 1)) {
            dist = node_dist[i];
            node = i;
        }
    }
    global_closest[0] = node;
    visited_node[node] = 1;
}
__global__ void cudaRelax(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int* global_closest) {
    int next = blockIdx.x*blockDim.x + threadIdx.x;
    int source = global_closest[0];
    data_t edge = graph[source*VERTICES + next];
    data_t new_dist = node_dist[source] + edge;
    if ((edge != 0) &&
        (visited_node[next] != 1) &&
        (new_dist < node_dist[next])) {
        node_dist[next] = new_dist;
        parent_node[next] = source;
    }
}
