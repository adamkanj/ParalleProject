#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>        
#define VERTICES 16384          
#define DENSITY 32              
#define MAX_WEIGHT 100000       
#define INF_DIST 1000000000     
#define IMPLEMENTATIONS 2       
#define THREADS 4               
#define RAND_SEED 1234          
typedef float data_t;           
#define CPG 3.611
#define GIG 1000000000
void main() {
    srand(RAND_SEED);
    void setIntArrayValue(int* in_array, int array_size, int value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    void initializeGraphZero(data_t* graph, int num_vertices);
    void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices);
    void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start);               
    void dijkstraCPUParallel(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start);             
    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec start, end;                 
    struct timespec time_stamp[IMPLEMENTATIONS];
    int graph_size      = VERTICES*VERTICES*sizeof(data_t);     
    int int_array       = VERTICES*sizeof(int);                 
    int data_array      = VERTICES*sizeof(data_t);              
    data_t* graph       = (data_t*)malloc(graph_size);                
    data_t* node_dist   = (data_t*)malloc(data_array);                
    int* parent_node    = (int*)malloc(int_array);                    
    int* edge_count     = (int*)malloc(int_array);                    
    int* visited_node   = (int*)malloc(int_array);                    
    int *pn_matrix      = (int*)malloc(IMPLEMENTATIONS*int_array);    
    data_t* dist_matrix = (data_t*)malloc(IMPLEMENTATIONS*data_array);
    setIntArrayValue(edge_count, VERTICES, 0);       
    initializeGraphZero(graph, VERTICES);            
    constructGraphEdge(graph, edge_count, VERTICES); 
    free(edge_count);                   
    int i;                                  
    int origin = (rand() % VERTICES);       
    printf("Origin vertex: %d\n\n", origin);
    int version = 0;
    printf("Running serial...");
    clock_gettime(CLOCK_REALTIME, &start);
    dijkstraCPUSerial(graph, node_dist, parent_node, visited_node, VERTICES, origin);
    clock_gettime(CLOCK_REALTIME, &end);
    time_stamp[version] = diff(start, end);         
    for (i = 0; i < VERTICES; i++) {                
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }printf("Done!\n");
    version++;
    printf("Running OpenMP...");
    clock_gettime(CLOCK_REALTIME, &start);
    dijkstraCPUParallel(graph, node_dist, parent_node, visited_node, VERTICES, origin);
    clock_gettime(CLOCK_REALTIME, &end);
    time_stamp[version] = diff(start, end);         
    for (i = 0; i < VERTICES; i++) {                
        pn_matrix[version*VERTICES + i] = parent_node[i];
        dist_matrix[version*VERTICES + i] = node_dist[i];
    }printf("Done!\n");
    printf("\nVertices: %d", VERTICES);
    printf("\nDensity: %d", DENSITY);
    printf("\nMax Weight: %d", MAX_WEIGHT);
    printf("\n\nTime (cycles):\nSerial,OpenMP\n");
    for (i = 0; i < IMPLEMENTATIONS; i++) {
        printf("%ld,", (long int)( (double)(CPG)*(double)
            (GIG * time_stamp[i].tv_sec + time_stamp[i].tv_nsec)));
    }printf("\n\nError checking:\n");
    printf("----Serial vs OPenMP:\n");
    int p_errors = 0, d_errors = 0;
    for (i = 0; i < VERTICES; i++) {
        if (pn_matrix[i] != pn_matrix[VERTICES + i]) {
            p_errors++;
        }
        if (dist_matrix[i] != dist_matrix[VERTICES + i]) {
            d_errors++;
        }
    }printf("--------%d parent errors found.\n", p_errors);
    printf("--------%d dist errors found.\n", d_errors);
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
    }return node;    
}
int closestNodeOMP(data_t* node_dist, int* visited_node, int num_vertices) {
    data_t min_dist = INF_DIST + 1;  
    int min_node = -1;              
    int min_dist_thread, min_node_thread;
    int vertex;
    omp_set_num_threads(THREADS);
    #pragma omp parallel private(min_dist_thread, min_node_thread) shared(node_dist, visited_node){
        min_dist_thread = min_dist;             
        min_node_thread = min_node;             
        #pragma omp barrier                     
        #pragma omp for nowait                      
        for (vertex = 0; vertex < num_vertices; vertex++) {            
            if ((node_dist[vertex] < min_dist_thread) && (visited_node[vertex] == 0)) {
                min_dist_thread = node_dist[vertex];
                min_node_thread = vertex;
            }
        }#pragma omp critical{
            if (min_dist_thread < min_dist) {
                min_dist = min_dist_thread;
                min_node = min_node_thread;
            }
        }
    }return min_node;
}
void constructGraphEdge(data_t* graph, int* edge_count, int num_vertices) {
    int i;                  
    int rand_vertex;        
    int curr_num_edges;     
    int num_edges;          
    data_t edge, weight;    
    for (i = 1; i < num_vertices; i++) {
        rand_vertex = (rand() % i);                     
        weight = (rand() % MAX_WEIGHT) + 1;             
        graph[rand_vertex*num_vertices + i] = weight;   
        graph[i*num_vertices + rand_vertex] = weight;
        edge_count[i] += 1;                             
        edge_count[rand_vertex] += 1;
    }for (i = 0; i < num_vertices; i++) {    
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
void checkArray(int* a, int length) {
    int i;
    printf("Proof: ");
    for (i = 0; i < length; i++) {
        printf("%d, ", a[i]);
    }printf("\n");
}
struct timespec diff(struct timespec start, struct timespec end){
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }return temp;
}
void dijkstraCPUSerial(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start) {
    void setIntArrayValue(int* in_array, int array_size, int init_value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);
    void checkArray(int* a, int length);
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
void dijkstraCPUParallel(data_t* graph, data_t* node_dist, int* parent_node, int* visited_node, int num_vertices, int v_start) {
    void setIntArrayValue(int* in_array, int array_size, int init_value);
    void setDataArrayValue(data_t* in_array, int array_size, data_t init_value);
    int closestNodeOMP(data_t* node_dist, int* visited_node, int num_vertices);
    int closestNode(data_t* node_dist, int* visited_node, int num_vertices);
    setDataArrayValue(node_dist, VERTICES, INF_DIST);     
    setIntArrayValue(parent_node, VERTICES, -1);          
    setIntArrayValue(visited_node, VERTICES, 0);          
    node_dist[v_start] = 0;                     
    int i, next;
    for (i = 0; i < num_vertices; i++) {
        int curr_node = closestNodeOMP(node_dist, visited_node, num_vertices);
        visited_node[curr_node] = 1;
        omp_set_num_threads(THREADS);
        int new_dist;
        #pragma omp parallel shared(graph,node_dist) {
            #pragma omp for private(new_dist,next)
            for (next = 0; next < num_vertices; next++) {
                new_dist = node_dist[curr_node] + graph[curr_node*num_vertices + next];
                if ((visited_node[next] != 1)                                  
                    && (graph[curr_node*num_vertices + next] != (data_t)(0))   
                    && (new_dist < node_dist[next])) {                         
                    node_dist[next] = new_dist;        
                    parent_node[next] = curr_node;    
                }
            }#pragma omp barrier
        }
    }
}
