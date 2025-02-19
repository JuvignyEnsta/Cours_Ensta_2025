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
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Paramètres de l'image
width, height = 1024, 1024
scaleX = 3.0 / width
scaleY = 2.25 / height
max_iterations = 100

mandelbrot_set = MandelbrotSet(max_iterations=max_iterations, escape_radius=10)

if rank == 0:
    # ---- PROCESSUS MAÎTRE ----
    start_time = time.time()
    # Liste des tâches (numéros de lignes à traiter)
    task_list = list(range(height))
    num_workers = size - 1
    active_workers = num_workers

    # Stockage des résultats
    result_image = np.zeros((height, width), dtype=np.float64)

    # Envoi initial des premières tâches
    for worker in range(1, num_workers + 1):
        if task_list:
            task = task_list.pop(0)
            comm.send(task, dest=worker, tag=1)
        else:
            comm.send(None, dest=worker, tag=2)  # Pas de tâches restantes
            active_workers -= 1

    # Réception et distribution des tâches
    while active_workers > 0:
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()
        row, values = data

        # Stocker les résultats
        result_image[row, :] = values

        # Envoyer une nouvelle tâche s'il en reste
        if task_list:
            task = task_list.pop(0)
            comm.send(task, dest=worker, tag=1)
        else:
            comm.send(None, dest=worker, tag=2)  # Plus de travail -> fin
            active_workers -= 1

    end_time = time.time()
    print(f"Temps d'exécution (Master-Worker, {size} processus) : {end_time - start_time:.4f} s")

    # Sauvegarde de l'image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(result_image) * 255))
    image.save("mandelbrot_master_worker.png")

else:
    # ---- PROCESSUS ESCLAVE ----
    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG)
        if task is None:
            break  # Fin des tâches

        # Calculer la ligne demandée
        row_values = np.zeros(width, dtype=np.float64)
        y = -1.125 + scaleY * task
        for x in range(width):
            c = complex(-2.0 + scaleX * x, y)
            row_values[x] = mandelbrot_set.convergence(c, smooth=True)

        # Envoyer les résultats au maître
        comm.send((task, row_values), dest=0, tag=3)
