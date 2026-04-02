"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. 
...
Version parallélisée avec mpi4py (2 processus : 1 pour le calcul, 1 pour l'affichage)
"""
import pygame as pg
import numpy as np
from mpi4py import MPI
import time
import sys


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
            self.cells[indices_i,indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []
        for i in range(ny):
            i_above = (i+ny-1)%ny
            i_below = (i+1)%ny
            for j in range(nx):
                j_left = (j-1+nx)%nx
                j_right= (j+1)%nx
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                voisines = np.array(self.cells[voisins_i,voisins_j])
                nb_voisines_vivantes = np.sum(voisines)
                if self.cells[i,j] == 1: 
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i,j] = 0 
                        diff_cells.append(i*nx+j)
                    else:
                        next_cells[i,j] = 1 
                elif nb_voisines_vivantes == 3: 
                    next_cells[i,j] = 1         
                    diff_cells.append(i*nx+j)
                else:
                    next_cells[i,j] = 0         
        self.cells = next_cells
        return diff_cells


class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[1]//grid.dimensions[1]
        self.size_y = geometry[0]//grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4 :
            self.draw_color=pg.Color('lightgrey')
        else:
            self.draw_color=None
        self.width = grid.dimensions[1] * self.size_x
        self.height= grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width,self.height))
        self.canvas_cells = []

    def compute_rectangle(self, i: int, j: int):
        return (self.size_x*j, self.height - self.size_y*(i + 1), self.size_x, self.size_y)

    def compute_color(self, i: int, j: int):
        if self.grid.cells[i,j] == 0:
            return self.grid.col_dead
        else:
            return self.grid.col_life

    def draw(self):
        [self.screen.fill(self.compute_color(i,j),self.compute_rectangle(i,j)) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        if (self.draw_color is not None):
            [pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y)) for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height)) for j in range(self.grid.dimensions[1])]
        pg.display.update()


if __name__ == '__main__':
    # --- INITIALISATION MPI ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Vérification stricte du nombre de processus
    if size != 2:
        if rank == 0:
            print("Erreur : Ce script nécessite exactement 2 processus.")
            print("Lancez-le avec : mpiexec -n 2 python game_of_life.py")
        sys.exit(1)

    dico_patterns = { 
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    
    choice = "glider"
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
        
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        if rank == 0:
            print("No such pattern. Available ones are:", dico_patterns.keys())
        sys.exit(1)

    if rank == 0:
        if rank == 0:
            print(f"Pattern initial choisi : {choice}")
            print(f"resolution ecran : {resx,resy}")
            print("Processus 0 : En charge des calculs")
            print("Processus 1 : En charge de l'affichage")
            
    # ========================================================
    # LOGIQUE PROCESSUS 0 (CALCUL SEUL)
    # ========================================================
    if rank == 0:
        grid = Grille(*init_pattern)
        mustContinue = True
        
        while mustContinue:
            time.sleep(0.5) # A régler ou commenter pour vitesse maxi
            t1 = time.time()
            # 1. Calcule la génération et récupère la liste des différences
            diff = grid.compute_next_iteration()
            t2 = time.time()
            
            # 2. Envoie les différences au processus 1
            comm.send(diff, dest=1, tag=11)
            
            # 3. Reçoit l'ordre de continuer (ou s'arrêter si on a fermé la fenêtre)
            mustContinue = comm.recv(source=1, tag=22)
            
            print(f"P0 Calcul : {t2-t1:2.2e} s")
    # ========================================================
    # LOGIQUE PROCESSUS 1 (AFFICHAGE SEUL)
    # ========================================================
    elif rank == 1:
        pg.init()
        # On initialise la grille à l'identique pour avoir le même point de départ
        grid = Grille(*init_pattern)
        appli = App((resx, resy), grid)
        mustContinue = True
        nx = grid.dimensions[1] # Nécessaire pour repasser d'un index 1D à 2D
        
        while mustContinue:
            # 1. Attente et réception des modifications envoyées par le P0
            diff = comm.recv(source=0, tag=11)
            
            # 2. Mise à jour de la grille locale (inversion des états)
            for index in diff:
                i = index // nx
                j = index % nx
                grid.cells[i, j] ^= 1 # ^ 1 inverse l'état (0 devient 1, 1 devient 0)
                
            # 3. Affichage
            t2 = time.time()
            appli.draw()
            t3 = time.time()
            print(f"P1 Affichage : {t3-t2:2.2e} s")
            # 4. Gestion des événements pygame
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
                    
            # 5. Envoi du signal au Processus 0 pour l'autoriser à calculer la suite
            comm.send(mustContinue, dest=0, tag=22)

        pg.quit()
