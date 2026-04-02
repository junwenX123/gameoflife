"""
Simulateur d'Automate Cellulaire (Conway's Game of Life)
Parallélisation par Blocs 2D avec MPI (Non-blocking & Advanced Overlap)
"""
import sys
import time
import pygame as pg
import numpy as np
from mpi4py import MPI


class PygameRenderer:
    """
    Moteur de rendu graphique utilisant pygame.surfarray pour des performances optimales.
    S'exécute uniquement sur le processus Maître (Rank 0).
    """
    def __init__(self, window_size, universe_shape, color_alive=pg.Color("black"), color_empty=pg.Color("white")):
        self.universe_shape = universe_shape
        self.cell_width = window_size[1] // universe_shape[1]
        self.cell_height = window_size[0] // universe_shape[0]
        
        self.total_w = universe_shape[1] * self.cell_width
        self.total_h = universe_shape[0] * self.cell_height
        self.display_surface = pg.display.set_mode((self.total_w, self.total_h))
        
        self.c_alive = color_alive
        self.c_empty = color_empty

        # Définition de la grille visuelle
        self.grid_lines_surface = None
        if self.cell_width > 4 and self.cell_height > 4:
            self.line_color = pg.Color('lightgrey')
            self.grid_lines_surface = pg.Surface((self.total_w, self.total_h), pg.SRCALPHA)
            for row in range(self.universe_shape[0]):
                pg.draw.line(self.grid_lines_surface, self.line_color, (0, row * self.cell_height), (self.total_w, row * self.cell_height))
            for col in range(self.universe_shape[1]):
                pg.draw.line(self.grid_lines_surface, self.line_color, (col * self.cell_width, 0), (col * self.cell_width, self.total_h))

    def render_frame(self, full_grid_state):
        """Met à jour l'affichage en convertissant la matrice NumPy en surface."""
        color_array = np.full((self.universe_shape[0], self.universe_shape[1], 3), self.c_empty[:3], dtype=np.uint8)
        color_array[full_grid_state == 1] = self.c_alive[:3]
        
        # Adaptation des axes pour Pygame (X, Y)
        surface_array = np.swapaxes(color_array, 0, 1)
        surface_array = surface_array[:, ::-1, :] 
        
        fast_surface = pg.surfarray.make_surface(surface_array)
        fast_surface = pg.transform.scale(fast_surface, (self.total_w, self.total_h))
        self.display_surface.blit(fast_surface, (0, 0))
        
        if self.grid_lines_surface:
            self.display_surface.blit(self.grid_lines_surface, (0, 0))
            
        pg.display.update()


class SubBlockGrid:
    """
    Sous-grille gérant un bloc 2D. 
    Intègre une logique de communication en 2 phases pour transporter les coins implicitement.
    """
    def __init__(self, worker_id, p_rows, p_cols, shape, seed_coords=None):
        self.my_id = worker_id
        global_h, global_w = shape
        
        grid_row = worker_id // p_cols
        grid_col = worker_id % p_cols

        # Partitionnement 2D
        chunk_y = global_h // p_rows
        rest_y = global_h % p_rows
        self.local_h = chunk_y + (1 if grid_row < rest_y else 0)
        self.offset_y = grid_row * chunk_y + min(grid_row, rest_y)

        chunk_x = global_w // p_cols
        rest_x = global_w % p_cols
        self.local_w = chunk_x + (1 if grid_col < rest_x else 0)
        self.offset_x = grid_col * chunk_x + min(grid_col, rest_x)

        # Matrice locale avec halos (+2 en hauteur et largeur)
        self.matrix = np.zeros((self.local_h + 2, self.local_w + 2), dtype=np.uint8)
        self.future_matrix = np.zeros_like(self.matrix)

        if seed_coords:
            for (y_glb, x_glb) in seed_coords:
                if (self.offset_y <= y_glb < self.offset_y + self.local_h and
                    self.offset_x <= x_glb < self.offset_x + self.local_w):
                    self.matrix[y_glb - self.offset_y + 1, x_glb - self.offset_x + 1] = 1
        else:
            self.matrix[1:self.local_h+1, 1:self.local_w+1] = np.random.randint(2, size=(self.local_h, self.local_w), dtype=np.uint8)

        # Identification des voisins toriques
        self.neighbor_up = ((grid_row - 1) % p_rows) * p_cols + grid_col
        self.neighbor_down = ((grid_row + 1) % p_rows) * p_cols + grid_col
        self.neighbor_left = grid_row * p_cols + (grid_col - 1) % p_cols
        self.neighbor_right = grid_row * p_cols + (grid_col + 1) % p_cols

    def _compute_area(self, r_start, r_end, c_start, c_end):
        """Moteur de calcul interne"""
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                voisines = [
                    self.matrix[i-1, j-1], self.matrix[i-1, j], self.matrix[i-1, j+1],
                    self.matrix[i,   j-1],                      self.matrix[i,   j+1],
                    self.matrix[i+1, j-1], self.matrix[i+1, j], self.matrix[i+1, j+1]
                ]
                nb = sum(voisines)
                state = self.matrix[i, j]
                if state == 1:
                    self.future_matrix[i, j] = 1 if (nb == 2 or nb == 3) else 0
                else:
                    self.future_matrix[i, j] = 1 if nb == 3 else 0

    def step_generation_async(self, comm):
        """
        Logique avancée : Échange en 2 phases avec buffers contigus ET TAGS stricts.
        """
        if comm.Get_size() == 1:
            # Synchronisation locale circulaire
            self.matrix[0, :] = self.matrix[-2, :]
            self.matrix[-1, :] = self.matrix[1, :]
            self.matrix[:, 0] = self.matrix[:, -2]
            self.matrix[:, -1] = self.matrix[:, 1]
            self._compute_area(1, self.local_h + 1, 1, self.local_w + 1)
            self.matrix, self.future_matrix = self.future_matrix, self.matrix
            return

        # ================= PHASE 1 : Verticale =================
        reqs_v = [
            # CRÉATION DES TAGS : 2 pour ce qui descend, 1 pour ce qui monte
            comm.Irecv(self.matrix[0, 1:-1], source=self.neighbor_up, tag=2),
            comm.Irecv(self.matrix[-1, 1:-1], source=self.neighbor_down, tag=1),
            comm.Isend(self.matrix[1, 1:-1].copy(), dest=self.neighbor_up, tag=1),
            comm.Isend(self.matrix[-2, 1:-1].copy(), dest=self.neighbor_down, tag=2)
        ]

        # OVERLAP 1 : Calcul du noyau central 
        self._compute_area(2, self.local_h, 2, self.local_w)

        # Attente des données verticales
        MPI.Request.Waitall(reqs_v)

        # ================= PHASE 2 : Horizontale (Inclus les coins) =================
        recv_left_col = np.empty(self.local_h + 2, dtype=np.uint8)
        recv_right_col = np.empty(self.local_h + 2, dtype=np.uint8)

        reqs_h = [
            # CRÉATION DES TAGS : 4 pour ce qui va vers la droite, 3 pour ce qui va vers la gauche
            comm.Irecv(recv_left_col, source=self.neighbor_left, tag=4),
            comm.Irecv(recv_right_col, source=self.neighbor_right, tag=3),
            comm.Isend(self.matrix[:, 1].copy(), dest=self.neighbor_left, tag=3),
            comm.Isend(self.matrix[:, -2].copy(), dest=self.neighbor_right, tag=4)
        ]

        # OVERLAP 2 : Calcul des frontières Haut/Bas
        self._compute_area(1, 2, 2, self.local_w) # Top
        if self.local_h > 1:
            self._compute_area(self.local_h, self.local_h + 1, 2, self.local_w) # Bottom

        # Attente des données horizontales
        MPI.Request.Waitall(reqs_h)

        # Re-assignation manuelle des colonnes
        self.matrix[:, 0] = recv_left_col
        self.matrix[:, -1] = recv_right_col

        # OVERLAP 3 : Calcul des frontières Gauche/Droite
        self._compute_area(1, self.local_h + 1, 1, 2) # Left
        if self.local_w > 1:
            self._compute_area(1, self.local_h + 1, self.local_w, self.local_w + 1) # Right

        # Bascule d'état
        self.matrix, self.future_matrix = self.future_matrix, self.matrix

class SimulationController:
    """
    Contrôleur principal orchestrant la simulation en Blocs 2D.
    """
    AVAILABLE_SEEDS = { 
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

    def __init__(self):
        self.global_comm = MPI.COMM_WORLD.Dup()
        self.process_rank = self.global_comm.Get_rank()
        self.total_processes = self.global_comm.Get_size()
        
        # Split logic (Master vs Workers)
        split_color = 0 if self.process_rank == 0 else 1
        self.workers_comm = self.global_comm.Split(split_color, self.process_rank)
        if self.process_rank == 0:
            self.workers_comm = None

        self._parse_args_and_setup()
        self.is_running = True

    def _parse_args_and_setup(self):
        target_pattern = 'glider' if len(sys.argv) <= 1 else sys.argv[1]
        res_w, res_h = 800, 800
        if len(sys.argv) > 3:
            res_w, res_h = int(sys.argv[2]), int(sys.argv[3])

        self.universe_dim, seed_data = self.AVAILABLE_SEEDS[target_pattern]
        self.is_standalone = (self.total_processes == 1)

        # Initialisation Maître
        if self.process_rank == 0:
            pg.init()
            print(f"[*] Sim-2D-Block: Pattern={target_pattern}, Res={res_w}x{res_h}")
            self.renderer = PygameRenderer((res_w, res_h), self.universe_dim)
        else:
            self.renderer = None

        # Calcul inline de la topologie 2D optimale
        worker_nbp = self.total_processes - 1 if not self.is_standalone else 1
        self.p_rows, self.p_cols = 1, worker_nbp
        min_diff = worker_nbp
        for r in range(1, int(worker_nbp**0.5) + 1):
            if worker_nbp % r == 0:
                c = worker_nbp // r
                if abs(c - r) < min_diff:
                    self.p_rows, self.p_cols = r, c
                    min_diff = abs(c - r)

        if self.process_rank == 0 and not self.is_standalone:
            print(f"Grille de calcul 2D: {self.p_rows}x{self.p_cols} (Total {worker_nbp} workers)")

        # Instanciation de la grille locale
        self.local_grid = None
        if self.is_standalone:
            self.local_grid = SubBlockGrid(0, 1, 1, self.universe_dim, seed_data)
        elif self.process_rank != 0:
            self.local_grid = SubBlockGrid(self.process_rank - 1, self.p_rows, self.p_cols, self.universe_dim, seed_data)

        # Pré-calcul des paramètres Gather
        self._setup_mpi_gathering()

    def _setup_mpi_gathering(self):
        """Pré-calcule les buffers Gatherv une seule fois avant la boucle"""
        if self.process_rank != 0:
            info = np.array([self.local_grid.local_h, self.local_grid.local_w, 
                             self.local_grid.offset_y, self.local_grid.offset_x], dtype=int)
        else:
            info = np.array([0, 0, 0, 0], dtype=int)

        self.all_info = np.zeros((self.total_processes, 4), dtype=int) if self.process_rank == 0 else None
        self.global_comm.Gather(info, self.all_info, root=0)

        if self.process_rank == 0:
            self.recvcounts = np.zeros(self.total_processes, dtype=int)
            for i in range(1, self.total_processes):
                self.recvcounts[i] = self.all_info[i, 0] * self.all_info[i, 1]
            self.displs = np.insert(np.cumsum(self.recvcounts)[:-1], 0, 0)
            self.global_flat = np.zeros(np.sum(self.recvcounts), dtype=np.uint8)
        else:
            self.recvcounts, self.displs, self.global_flat = None, None, None

    def execute_simulation(self):
        while self.is_running:
            # === ETAPE 1 : Calcul Local (Overlap Async) ===
            if self.is_standalone:
                t_calc_start = time.time()
                self.local_grid.step_generation_async(self.global_comm)
                compute_time = time.time() - t_calc_start
                local_payload = None
            else:
                if self.process_rank == 0: t_master_wait = time.time()
                
                if self.process_rank != 0:
                    t_calc_start = time.time()
                    self.local_grid.step_generation_async(self.workers_comm)
                    compute_time = time.time() - t_calc_start
                    local_payload = self.local_grid.matrix[1:-1, 1:-1].flatten()
                else:
                    compute_time, local_payload = 0.0, np.array([], dtype=np.uint8)

                # === ETAPE 2 : Rassemblement Gatherv ===
                self.global_comm.Gatherv(local_payload, [self.global_flat, self.recvcounts, self.displs, MPI.UINT8_T], root=0)
                if self.process_rank == 0: compute_time = time.time() - t_master_wait

            # === ETAPE 3 : Rendu Graphique (Rank 0) ===
            if self.is_standalone:
                t_render_start = time.time()
                self.renderer.render_frame(self.local_grid.matrix[1:-1, 1:-1])
                render_time = time.time() - t_render_start
                self._handle_events()
                print(f"Sync-2D | Calc: {compute_time:.2e}s | Rendu: {render_time:.2e}s\r", end='')
            else:
                if self.process_rank == 0:
                    final_grid = np.zeros(self.universe_dim, dtype=np.uint8)
                    for k in range(1, self.total_processes):
                        ny, nx, y_start, x_start = self.all_info[k]
                        bloc = self.global_flat[self.displs[k]:self.displs[k]+self.recvcounts[k]].reshape((ny, nx))
                        final_grid[y_start:y_start+ny, x_start:x_start+nx] = bloc

                    t_render_start = time.time()
                    self.renderer.render_frame(final_grid)
                    render_time = time.time() - t_render_start
                    self._handle_events()
                    print(f"Workers-2D: {compute_time:.2e}s | Master: {render_time:.2e}s\r", end='')

            self.is_running = self.global_comm.bcast(self.is_running, root=0)

        if self.process_rank == 0: pg.quit()

    def _handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT: self.is_running = False


if __name__ == '__main__':
    sim_app = SimulationController()
    sim_app.execute_simulation()