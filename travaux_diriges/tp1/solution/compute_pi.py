# Calcul pi par une méthode stochastique (convergence très lente !)
import time
import numpy as np
from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank
filename = f"Output{rank:03d}.txt"
out      = open(filename, mode='w')

# Nombre d'échantillons :
nbSamples = 40000000

nbSamples_loc = nbSamples//nbp
reste_samples = nbSamples%nbp
if rank < reste_samples: nbSamples_loc += 1


beg = time.time()
# Tirage des points (x,y) tirés dans un carré [-1;1] x [-1; 1]
x = 2.*np.random.random_sample((nbSamples_loc,))-1.
y = 2.*np.random.random_sample((nbSamples_loc,))-1.
# Création masque pour les points dans le cercle unité
filtre = np.array(x*x+y*y<1.)
# Compte le nombre de points dans le cercle unité
sum = np.add.reduce(filtre, 0)

approx_pi_loc = np.array([4.*sum/nbSamples], dtype=np.double)
approx_pi_glob= np.zeros(1, dtype=np.double)
globCom.Allreduce(approx_pi_loc, approx_pi_glob, MPI.SUM)
end = time.time()

out.write(f"Temps pour calculer pi : {end - beg} secondes\n")
out.write(f"Pi vaut environ {approx_pi_glob[0]}\n")

out.close()
