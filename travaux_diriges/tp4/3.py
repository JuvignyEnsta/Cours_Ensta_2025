"""
Le jeu de la vie avec MPI pour la décomposition de domaine + séparation calcul/affichage
"""
import pygame as pg
import numpy as np
from mpi4py import MPI
import time  

# Initialisation de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    """
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self, start_row, end_row):
        """
        Calcule la prochaine génération de cellules pour une partie de la grille (lignes start_row à end_row)
        """
        partial_cells = self.cells[start_row:end_row, :]
        neighbours_count = sum(
            np.roll(np.roll(partial_cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0)
        )
        next_cells = (neighbours_count == 3) | (partial_cells & (neighbours_count == 2))
        return next_cells

class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.colors = np.array([self.grid.col_dead[:-1], self.grid.col_life[:-1]])

    def draw(self, cells):
        surface = pg.surfarray.make_surface(self.colors[cells.T])
        surface = pg.transform.flip(surface, False, True)
        surface = pg.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))
        if self.draw_color is not None:
            [pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y)) for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height)) for j in range(self.grid.dimensions[1])]
        pg.display.update()


if __name__ == '__main__':
    # Initialisation de pygame
    pg.init()

    dico_patterns = {
        'glider': ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)])
    }

    choice = 'glider'
    resx, resy = 800, 800
    init_pattern = dico_patterns[choice]

    grid = Grille(*init_pattern)

    # Décomposition de la grille par rang
    rows_per_rank = grid.dimensions[0] // size
    start_row = rank * rows_per_rank
    end_row = (rank + 1) * rows_per_rank if rank != size - 1 else grid.dimensions[0]

    if rank == 0:

        appli = App((resx, resy), grid)
        loop = True
        while loop:
            all_cells = np.zeros(grid.dimensions, dtype=np.uint8)


            all_cells[start_row:end_row, :] = grid.compute_next_iteration(start_row, end_row)

            for r in range(1, size):
                if comm.Iprobe(source=r):  
                    data = comm.recv(source=r)
                    all_cells[data['start_row']:data['end_row'], :] = data['cells']

            grid.cells = all_cells
            appli.draw(all_cells)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    loop = False
                    for r in range(1, size):
                        comm.send({'stop': True}, dest=r)

    else:
        loop = True
        while loop:
            next_cells = grid.compute_next_iteration(start_row, end_row)
            comm.send({'cells': next_cells, 'start_row': start_row, 'end_row': end_row}, dest=0)

            time.sleep(0.1)

            if comm.Iprobe(source=0):
                stop_signal = comm.recv(source=0)
                if 'stop' in stop_signal:
                    loop = False

    MPI.Finalize()

    if rank == 0:
        pg.quit()

