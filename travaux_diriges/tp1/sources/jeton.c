#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank; // Rang du processus
    int nbp;  // Nombre total de processus
    int jeton; // Valeur du jeton

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    if (nbp < 2) {
        if (rank == 0) {
            printf("Erreur : Ce programme nécessite au moins 2 processus MPI.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        jeton = 1;
        printf("Le processus %d initialise le jeton avec la valeur %d et l'envoie au processus 1.\n", rank, jeton);

        MPI_Ssend(&jeton, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&jeton, 1, MPI_INT, nbp - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Le processus %d reçoit le jeton final avec la valeur %d.\n", rank, jeton);
    } else {
        MPI_Recv(&jeton, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Le processus %d reçoit le jeton avec la valeur %d depuis le processus %d.\n", rank, jeton, rank - 1);

        jeton++;

        int nextp = (rank + 1) % nbp; // Calcul du rang du processus suivant
        MPI_Ssend(&jeton, 1, MPI_INT, nextp, 0, MPI_COMM_WORLD);
        printf("Le processus %d envoie le jeton avec la valeur %d au processus %d.\n", rank, jeton, nextp);
    }

    MPI_Finalize();
    return 0;
}
