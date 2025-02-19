from mpi4py import MPI
import numpy as np
from PIL import Image
import matplotlib.cm
from math import log
import time

class MandelbrotSet:
    def __init__(self, max_iterations=50, escape_radius=10):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def convergence(self, c: complex, smooth=True) -> float:
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

# Définition des paramètres de l'image
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height

# Meilleure répartition statique : Distribution des lignes de manière équilibrée
num_rows = height // nbp
remainder = height % nbp

if rank < remainder:
    start_row = rank * (num_rows + 1)
    end_row = start_row + num_rows + 1
else:
    start_row = rank * num_rows + remainder
    end_row = start_row + num_rows

local_convergence = np.empty((end_row - start_row, width), dtype=np.float64)

start_time = time.time()
for i, y in enumerate(range(start_row, end_row)):
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        local_convergence[i, x] = MandelbrotSet(max_iterations=100, escape_radius=10).convergence(c, smooth=True)
end_time = time.time()
local_time = end_time - start_time

# Collecte des résultats sur le processus maître
if rank == 0:
    global_convergence = np.empty((height, width), dtype=np.float64)
else:
    global_convergence = None

comm.Gather(local_convergence, global_convergence, root=0)

# Enregistrement de l'image sur le processus maître
if rank == 0:
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_convergence) * 255))
    image.save("mandelbrot_mpi_optimized.png")
    print(f"Temps d'exécution (nbp={nbp}) : {local_time:.4f} s")
