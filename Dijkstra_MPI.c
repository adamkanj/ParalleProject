#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#define N 2048
#define SOURCE 1
#define MAXINT 9999999
void SingleSource(int n, int source, int *wgt, int *lengths, MPI_Comm comm) {
    int temp[N];
    int i, j;
    int nlocal; 
    int *marker;
    int firstvtx;
    int lastvtx; 
    int u, udist;
    int lminpair[2], gminpair[2];
    int npes, myrank;
    MPI_Status status;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);
    nlocal = n / npes;
    firstvtx = myrank*nlocal;
    lastvtx = firstvtx + nlocal - 1;
    for (j = 0; j<nlocal; j++) {
        lengths[j] = wgt[source*nlocal + j];
    }
    marker = (int *)malloc(nlocal*sizeof(int));
    for (j = 0; j<nlocal; j++) {
        marker[j] = 1;
    }
    if (source >= firstvtx && source <= lastvtx) {
        marker[source - firstvtx] = 0;
    }
    for (i = 1; i<n; i++) {
        lminpair[0] = MAXINT;
        lminpair[1] = -1;
        for (j = 0; j<nlocal; j++) {
            if (marker[j] && lengths[j] < lminpair[0]) {
                lminpair[0] = lengths[j];
                lminpair[1] = firstvtx + j;
            }
        }
        MPI_Allreduce(lminpair, gminpair, 1, MPI_2INT, MPI_MINLOC, comm);
        udist = gminpair[0];
        u = gminpair[1];
        if (u == lminpair[1]) {
            marker[u - firstvtx] = 0;
        }
        for (j = 0; j<nlocal; j++) {
            if (marker[j] && ((udist + wgt[u*nlocal + j]) < lengths[j])) {
                lengths[j] = udist + wgt[u*nlocal + j];
            }
        }
    }
    free(marker);
}
int main(int argc, char *argv[]) {
    int npes, myrank, nlocal;
    int weight[N][N]; 
    int distance[N]; 
    int *localWeight; 
    int *localDistance;
    int sendbuf[N*N]; 
    int i, j, k;
    char fn[255];
    FILE *fp;
    double time_start, time_end;
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    nlocal = N/npes; 
    localWeight = (int *)malloc(nlocal*N*sizeof(int));
    localDistance = (int *)malloc(nlocal*sizeof(int));
    if (myrank == SOURCE) {
        strcpy(fn,"input2048.txt");
        fp = fopen(fn,"r");
        if ((fp = fopen(fn,"r")) == NULL) {
            printf("Can't open the input file: %s\n\n", fn);
            exit(1);
        }
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                fscanf(fp,"%d", &weight[i][j]);
            }
        }
        for(k=0; k<npes; ++k) {
            for(i=0; i<N;++i) {
                for(j=0; j<nlocal;++j) {
                    sendbuf[k*N*nlocal+i*nlocal+j]=weight[i][k*nlocal+j];
                }
            }
        }
    }
    MPI_Scatter(sendbuf, nlocal*N, MPI_INT, localWeight, nlocal*N, MPI_INT, SOURCE, MPI_COMM_WORLD);
    SingleSource(N, SOURCE, localWeight, localDistance, MPI_COMM_WORLD);
    MPI_Gather(localDistance, nlocal, MPI_INT, distance, nlocal, MPI_INT, SOURCE, MPI_COMM_WORLD);
    if (myrank == SOURCE) {
        printf("Nodes: %d\n", N);
        gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
        printf("time cost is %1f\n", time_end - time_start);
    }
    free(localWeight);
    free(localDistance);
    MPI_Finalize();
    return 0;
}
