"""
Le jeu de la vie avec MPI pour la décomposition de domaine 
"""
import pygame as pg
import numpy as np
from mpi4py import MPI

# Initialisation de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    """
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
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
        # Créer une vue partielle de la grille pour les lignes assignées à ce rank
        partial_cells = self.cells[start_row:end_row, :]
        neighbours_count = sum(np.roll(np.roll(partial_cells, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))
        next_cells = (neighbours_count == 3) | (partial_cells & (neighbours_count == 2))
        diff_cells = (next_cells != partial_cells)
        return next_cells, diff_cells


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
    import time
    import sys

    # Initialisation de pygame
    pg.init()

    # Dictionnaire des patterns
    dico_patterns = {
        'blinker': ((5, 5), [(2, 1), (2, 2), (2, 3)]),
        'toad': ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
        "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
        "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
        "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
        "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
        "glider_gun": ((200, 100), [(51, 76), (52, 74), (52, 76), (53, 64), (53, 65), (53, 72), (53, 73), (53, 86), (53, 87), (54, 63), (54, 67), (54, 72), (54, 73), (54, 86), (54, 87), (55, 52), (55, 53), (55, 62), (55, 68), (55, 72), (55, 73), (56, 52), (56, 53), (56, 62), (56, 66), (56, 68), (56, 69), (56, 74), (56, 76), (57, 62), (57, 68), (57, 76), (58, 63), (58, 67), (59, 64), (59, 65)]),
        "space_ship": ((25, 25), [(11, 13), (11, 14), (12, 11), (12, 12), (12, 14), (12, 15), (13, 11), (13, 12), (13, 13), (13, 14), (14, 12), (14, 13)]),
        "die_hard": ((100, 100), [(51, 57), (52, 51), (52, 52), (53, 52), (53, 56), (53, 57), (53, 58)]),
        "pulsar": ((17, 17), [(2, 4), (2, 5), (2, 6), (7, 4), (7, 5), (7, 6), (9, 4), (9, 5), (9, 6), (14, 4), (14, 5), (14, 6), (2, 10), (2, 11), (2, 12), (7, 10), (7, 11), (7, 12), (9, 10), (9, 11), (9, 12), (14, 10), (14, 11), (14, 12), (4, 2), (5, 2), (6, 2), (4, 7), (5, 7), (6, 7), (4, 9), (5, 9), (6, 9), (4, 14), (5, 14), (6, 14), (10, 2), (11, 2), (12, 2), (10, 7), (11, 7), (12, 7), (10, 9), (11, 9), (12, 9), (10, 14), (11, 14), (12, 14)]),
        "floraison": ((40, 40), [(19, 18), (19, 19), (19, 20), (20, 17), (20, 19), (20, 21), (21, 18), (21, 19), (21, 20)]),
        "block_switch_engine": ((400, 400), [(201, 202), (201, 203), (202, 202), (202, 203), (211, 203), (212, 204), (212, 202), (214, 204), (214, 201), (215, 201), (215, 202), (216, 201)]),
        "u": ((200, 200), [(101, 101), (102, 102), (103, 102), (103, 101), (104, 103), (105, 103), (105, 102), (105, 101), (105, 105), (103, 105), (102, 105), (101, 105), (101, 104)]),
        "flat": ((200, 400), [(80, 200), (81, 200), (82, 200), (83, 200), (84, 200), (85, 200), (86, 200), (87, 200), (89, 200), (90, 200), (91, 200), (92, 200), (93, 200), (97, 200), (98, 200), (99, 200), (106, 200), (107, 200), (108, 200), (109, 200), (110, 200), (111, 200), (112, 200), (114, 200), (115, 200), (116, 200), (117, 200), (118, 200)])
    }

    # Choix du pattern
    choice = 'glider'
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    print(f"Pattern initial choisi : {choice}")
    print(f"resolution ecran : {resx, resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)

    # Création de la grille
    grid = Grille(*init_pattern)

    # Décomposition de la grille par rang
    rows_per_rank = grid.dimensions[0] // size
    start_row = rank * rows_per_rank
    end_row = (rank + 1) * rows_per_rank if rank != size - 1 else grid.dimensions[0]

    # Initialisation de l'affichage (chaque rank a sa propre fenêtre)
    appli = App((resx, resy), grid)

    loop = True
    while loop:
        # Calcul de la prochaine génération pour la partie assignée
        next_cells, _ = grid.compute_next_iteration(start_row, end_row)
        grid.cells[start_row:end_row, :] = next_cells

        # Affichage de la grille
        appli.draw(grid.cells)

        # Gestion des événements (fermeture de la fenêtre)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                loop = False

    # Fermeture de pygame
    pg.quit()