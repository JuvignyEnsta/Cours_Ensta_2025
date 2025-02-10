#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, data;
    double start_time, end_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Vérifier que le nombre de processus est une puissance de 2
    if ((size & (size - 1)) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Le nombre de processus doit être une puissance de 2\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {
        data = 42; // Initialise le jeton
    }

    int dim = 0;
    while ((1 << dim) < size) {
        dim++;
    }

    start_time = MPI_Wtime();
    for (int d = 0; d < dim; d++) {
        int partner = rank ^ (1 << d);
        if (rank < partner) {
            MPI_Send(&data, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&data, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    end_time = MPI_Wtime();

    printf("Processus %d a reçu le jeton avec la valeur %d\n", rank, data);
    if (rank == 0) {
        printf("Temps d'exécution pour %d processus : %f secondes\n", size, end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}