import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
import multiprocessing

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0))  # Clamp entre 0 et 1

    def count_iterations(self, c: complex, smooth=False) -> float:
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if -0.75 < c.real < 0.5:
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations

        z = 0
        for iter in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations

# Paramètres de l'image
width, height = 1024, 1024
scaleX, scaleY = 3.0 / width, 2.25 / height
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)

def worker(task_queue, result_queue):
    """
    Fonction exécutée par chaque esclave. 
    Prend une ligne, la calcule et envoie le résultat.
    """
    while True:
        y = task_queue.get()
        if y is None:
            break  # Fin du travail
        row = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
            row[x] = mandelbrot_set.convergence(c, smooth=True)
        result_queue.put((y, row))

def main(nbp):
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Création des processus esclaves
    processes = [multiprocessing.Process(target=worker, args=(task_queue, result_queue)) for _ in range(nbp)]
    
    for p in processes:
        p.start()

    # Envoi des tâches
    start_calc = time()
    for y in range(height):
        task_queue.put(y)

    # Ajout des messages de fin
    for _ in range(nbp):
        task_queue.put(None)

    # Collecte des résultats
    convergence = np.empty((width, height), dtype=np.double)
    for _ in range(height):
        y, row = result_queue.get()
        convergence[:, y] = row
    end_calc = time()

    print(f"Temps du calcul (Maître-Esclave) : {end_calc - start_calc:.3f} s")

    # Création de l'image
    start_img = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))
    end_img = time()
    
    print(f"Temps de constitution de l'image : {end_img - start_img:.3f} s")
    image.show()

    # Fermeture des processus
    for p in processes:
        p.join()

if __name__ == "__main__":
    nbp = min(multiprocessing.cpu_count(), 8)  # Ne pas dépasser 8 cœurs pour limiter l'overhead
    main(nbp)