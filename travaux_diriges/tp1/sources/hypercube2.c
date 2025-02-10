#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, data;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "Ce programme nécessite exactement 4 processus.\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {
        data = 42; // Initialise le jeton
    }

    int dim = 2;
    for (int d = 0; d < dim; d++) {
        int partner = rank ^ (1 << d);
        if (rank < partner) {
            MPI_Send(&data, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&data, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    printf("Processus %d a reçu le jeton avec la valeur %d\n", rank, data);

    MPI_Finalize();
    return 0;
}