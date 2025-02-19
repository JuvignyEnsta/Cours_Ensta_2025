from mpi4py import MPI
import numpy as np
from PIL import Image
from math import log
import time
import matplotlib.cm

class MandelbrotSet:
    def __init__(self, max_iterations: int, escape_radius: float = 2.):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def convergence(self, c: np.ndarray, smooth=True) -> np.ndarray:
        iter_counts = np.full(c.shape, self.max_iterations, dtype=np.float64)
        z = np.zeros(c.shape, dtype=np.complex128)
        mask = np.ones(c.shape, dtype=bool)

        for it in range(self.max_iterations):
            z[mask] = z[mask] * z[mask] + c[mask]
            has_diverged = np.abs(z) > self.escape_radius
            iter_counts[has_diverged & mask] = it
            mask &= ~has_diverged
            if not np.any(mask):
                break

        if smooth:
            iter_counts[has_diverged] += 1 - np.log(np.log(np.abs(z[has_diverged]))) / log(2)

        return iter_counts / self.max_iterations

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Paramètres de l'image
width, height = 1024, 1024
max_iterations = 200
escape_radius = 2.0

scaleX = 3.0 / width
scaleY = 2.25 / height

# Répartition des colonnes
cols_per_proc = width // size
start_col = rank * cols_per_proc
end_col = (rank + 1) * cols_per_proc if rank != size - 1 else width

# Calcul Mandelbrot pour les colonnes assignées
mandelbrot_set = MandelbrotSet(max_iterations=max_iterations, escape_radius=escape_radius)
local_convergence = np.zeros((height, cols_per_proc), dtype=np.float64)

start_time = time.time()
for x in range(start_col, end_col):
    c = np.array([complex(-2.0 + scaleX * x, -1.125 + scaleY * y) for y in range(height)])
    local_convergence[:, x - start_col] = mandelbrot_set.convergence(c, smooth=True)
end_time = time.time()

# Récolte des résultats
global_convergence = None
if rank == 0:
    global_convergence = np.zeros((height, width), dtype=np.float64)

comm.Gather(local_convergence, global_convergence, root=0)

# Processus 0 sauvegarde l'image
if rank == 0:
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(global_convergence) * 255))
    image.save("mandelbrot_parallel_columns.png")
    print(f"Temps de calcul ({size} processus) : {end_time - start_time:.4f} s")
