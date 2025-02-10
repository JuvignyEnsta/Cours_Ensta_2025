#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, data;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Ce programme nécessite exactement 2 processus.\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {
        data = 42; // Initialise le jeton à une valeur entière
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Processus %d a envoyé le jeton avec la valeur %d\n", rank, data);
    } else if (rank == 1) {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Processus %d a reçu le jeton avec la valeur %d\n", rank, data);
    }

    MPI_Finalize();
    return 0;
}