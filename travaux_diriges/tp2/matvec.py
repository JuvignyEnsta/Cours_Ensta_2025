from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Paramètres du problème ---
dim = 1024             # Dimension de la matrice (dim x dim)
Nloc = dim // size     # Nombre de lignes par processus (on suppose que dim est divisible par size)

# --- Détermination de la portion locale de la matrice ---
# Pour le processus de rang 'rank', les lignes vont de start_row à end_row-1
start_row = rank * Nloc
end_row = (rank + 1) * Nloc

# Construction de la portion locale de la matrice A.
# La matrice complète est définie par : A[i, j] = ((i+j) % dim) + 1.
# Pour le bloc de lignes du processus, i va de start_row à end_row-1 et j de 0 à dim-1.
rows = np.arange(start_row, end_row).reshape(Nloc, 1)  # dimensions (Nloc, 1)
cols = np.arange(dim).reshape(1, dim)                  # dimensions (1, dim)
A_local = ((rows + cols) % dim) + 1.0

# Construction du vecteur u (commun à tous les processus)
# u[j] = j + 1 pour j = 0, 1, ..., dim-1
u = np.array([j + 1. for j in range(dim)], dtype=float)

# --- Calcul du produit matrice-vecteur en parallèle ---
# Chaque processus calcule le produit pour ses lignes locales :
#   v_local[i] = somme_{j=0}^{dim-1} A[i,j] * u[j]
t_start = MPI.Wtime()
v_local = np.dot(A_local, u)  # v_local a pour taille (Nloc,)
t_local = MPI.Wtime() - t_start

# Rassemblement des résultats partiels afin d'obtenir le vecteur v complet sur tous les processus.
# On utilise MPI.Allgather pour que chaque processus reçoive l'ensemble des contributions.
v = np.empty(dim, dtype=float)
comm.Allgather([v_local, MPI.DOUBLE], [v, MPI.DOUBLE])

# Récupération du temps maximum parmi tous les processus pour représenter le temps global de l'opération parallèle.
t_global = comm.reduce(t_local, op=MPI.MAX, root=0)

# --- Calcul en séquentiel et évaluation du speed-up (réalisé sur le processus 0) ---
if rank == 0:
    # Construction complète de la matrice A et du vecteur u
    A_full = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
    u_full = np.array([i + 1. for i in range(dim)], dtype=float)
    
    # Mesure du temps de calcul séquentiel
    t_seq_start = time.time()
    v_seq = A_full.dot(u_full)
    t_seq_end = time.time()
    t_seq = t_seq_end - t_seq_start
    
    # Calcul du speedup
    speedup = t_seq / t_global if t_global > 0 else float('inf')
    
    # Affichage des résultats
    print("=== Produit matrice-vecteur par partitionnement par lignes ===")
    print("\nRésultat séquentiel :")
    print("v =", v_seq)
    print("Temps séquentiel : {:.6f} s".format(t_seq))
    
    print("\nRésultat parallèle :")
    print("v =", v)
    print("Temps parallèle (max sur les processus) : {:.6f} s".format(t_global))
    
    print("\nSpeedup obtenu : {:.2f}".format(speedup))
