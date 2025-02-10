# Calcul pi par une méthode stochastique (convergence très lente !)
import time
import numpy as np
from mpi4py import MPI

# Initialisation du contexte MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Nombre total d'échantillons
nb_samples = 40_000_000

# Nombre d'échantillons par processus
samples_per_process = nb_samples // size

beg = time.time()

# Tirage des points (x,y) tirés dans un carré [-1;1] x [-1; 1]
x = 2. * np.random.random_sample((samples_per_process,)) - 1.
y = 2. * np.random.random_sample((samples_per_process,)) - 1.

# Création masque pour les points dans le cercle unité
filtre = np.array(x * x + y * y < 1.)

# Compte le nombre de points dans le cercle unité
local_sum = np.add.reduce(filtre, 0)

# Réduction des sommes locales pour obtenir la somme globale
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    approx_pi = 4. * global_sum / nb_samples
    end = time.time()
    print(f"Temps pour calculer pi : {end - beg} secondes")
    print(f"Pi vaut environ {approx_pi}")