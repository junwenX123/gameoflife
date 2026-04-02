"""
Simulateur d'Automate Cellulaire (Conway's Game of Life)
Parallélisation par colonnes avec MPI (Non-blocking communications)
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

        # Définition de la grille visuelle (Off-screen buffer)
        self.grid_lines_surface = None
        if self.cell_width > 4 and self.cell_height > 4:
            self.line_color = pg.Color('lightgrey')
            self.grid_lines_surface = pg.Surface((self.total_w, self.total_h), pg.SRCALPHA)
            for row in range(self.universe_shape[0]):
                pg.draw.line(self.grid_lines_surface, self.line_color, (0, row * self.cell_height), (self.total_w, row * self.cell_height))
            for col in range(self.universe_shape[1]):
                pg.draw.line(self.grid_lines_surface, self.line_color, (col * self.cell_width, 0), (col * self.cell_width, self.total_h))

    def render_frame(self, full_grid_state):
        """Met à jour l'affichage avec conversion vectorisée via surfarray."""
        # Préparation de la matrice RGB (y, x transposé en x, y pour pygame)
        rgb_matrix = np.zeros((self.universe_shape[1], self.universe_shape[0], 3), dtype=np.uint8)
        grid_t = full_grid_state.T
        
        # Colorisation via masques (vectorisée)
        rgb_matrix[grid_t == 0] = self.c_empty[:3] 
        rgb_matrix[grid_t == 1] = self.c_alive[:3]

        # Rendu surfarray avec correction d'axe
        fast_surface = pg.surfarray.make_surface(rgb_matrix)
        fast_surface = pg.transform.flip(fast_surface, False, True)
        fast_surface = pg.transform.scale(fast_surface, (self.total_w, self.total_h))
        
        self.display_surface.blit(fast_surface, (0, 0))
        
        if self.grid_lines_surface:
            self.display_surface.blit(self.grid_lines_surface, (0, 0))
            
        pg.display.update()


class RegionSubGrid:
    """
    Sous-grille gérant un sous-ensemble vertical (colonnes) de l'univers.
    """
    def __init__(self, mpi_rank, total_workers, shape, seed_coords=None):
        self.my_rank = mpi_rank
        self.rows, global_w = shape
        
        # Partitionnement par colonnes (Column block distribution)
        chunk_size = global_w // total_workers
        remainder = global_w % total_workers
        self.local_w = chunk_size + (1 if mpi_rank < remainder else 0)
        self.offset_x = mpi_rank * chunk_size + min(mpi_rank, remainder)
        
        # Matrice locale (+2 pour les colonnes fantômes gauche et droite)
        self.matrix = np.zeros((self.rows, self.local_w + 2), dtype=np.uint8)

        # Initialisation du pattern
        if seed_coords:
            for (y_glb, x_glb) in seed_coords:
                if self.offset_x <= x_glb < self.offset_x + self.local_w:
                    x_loc = x_glb - self.offset_x + 1 
                    self.matrix[y_glb, x_loc] = 1
        else:
            self.matrix[:, 1:self.local_w+1] = np.random.randint(2, size=(self.rows, self.local_w), dtype=np.uint8)

    def communicate_boundaries(self, comm):
        """
        Synchronisation des halos (colonnes) via communications non-bloquantes.
        """
        cluster_size = comm.Get_size()
        my_id = comm.Get_rank()

        if cluster_size == 1:
            self.matrix[:, 0] = self.matrix[:, -2]
            self.matrix[:, -1] = self.matrix[:, 1]
            return

        neighbor_left = (my_id - 1) % cluster_size
        neighbor_right = (my_id + 1) % cluster_size

        # Buffers pour Isend/Irecv
        buf_recv_l = np.empty(self.rows, dtype=np.uint8)
        buf_recv_r = np.empty(self.rows, dtype=np.uint8)
        data_send_l = self.matrix[:, 1].copy()
        data_send_r = self.matrix[:, -2].copy()

        # Non-blocking Exchange
        mpi_requests = [
            comm.Irecv(buf_recv_l, source=neighbor_left),
            comm.Irecv(buf_recv_r, source=neighbor_right),
            comm.Isend(data_send_l, dest=neighbor_left),
            comm.Isend(data_send_r, dest=neighbor_right)
        ]
        
        MPI.Request.Waitall(mpi_requests)
        
        self.matrix[:, 0] = buf_recv_l
        self.matrix[:, -1] = buf_recv_r

    def step_generation(self):
        """
        Application des règles avec doubles boucles for (Parallélisation par colonnes).
        """
        h = self.rows
        active_cols = self.local_w
        future_matrix = np.zeros_like(self.matrix)

        for i in range(h):
            # Gestion du tore verticalement (Y)
            i_above = (i - 1 + h) % h
            i_below = (i + 1) % h
            
            for j in range(1, active_cols + 1):
                j_left = j - 1
                j_right = j + 1
                
                # Somme des 8 voisins
                total_neighbors = (
                    self.matrix[i_above, j_left] + self.matrix[i_above, j] + self.matrix[i_above, j_right] +
                    self.matrix[i, j_left] +                                 self.matrix[i, j_right] +
                    self.matrix[i_below, j_left] + self.matrix[i_below, j] + self.matrix[i_below, j_right]
                )

                # Logique standard
                current_state = self.matrix[i, j]
                if current_state == 1:
                    future_matrix[i, j] = 1 if (total_neighbors == 2 or total_neighbors == 3) else 0
                else:
                    future_matrix[i, j] = 1 if total_neighbors == 3 else 0
                    
        self.matrix = future_matrix


class SimulationController:
    """
    Contrôleur gérant l'orchestration globale de la simulation par colonnes.
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

        if self.process_rank == 0:
            pg.init()
            print(f"[*] Sim-Column: Pattern={target_pattern}, Res={res_w}x{res_h}")

        self.universe_dim, seed_data = self.AVAILABLE_SEEDS[target_pattern]
        self.is_standalone = (self.total_processes == 1)

        self.local_grid = None
        if self.is_standalone:
            self.local_grid = RegionSubGrid(0, 1, self.universe_dim, seed_data)
        elif self.process_rank != 0:
            self.local_grid = RegionSubGrid(self.process_rank - 1, self.total_processes - 1, self.universe_dim, seed_data)

        self.renderer = PygameRenderer((res_w, res_h), self.universe_dim) if self.process_rank == 0 else None

    def execute_simulation(self):
        while self.is_running:
            # === ETAPE 1 : Phase de Calcul ===
            if self.is_standalone:
                t_calc_start = time.time()
                self.local_grid.communicate_boundaries(self.global_comm)
                self.local_grid.step_generation()
                compute_time = time.time() - t_calc_start
                local_payload = None
                cols_count = self.local_grid.local_w
            else:
                if self.process_rank == 0: t_master_wait = time.time()
                
                if self.process_rank != 0:
                    t_calc_start = time.time()
                    self.local_grid.communicate_boundaries(self.workers_comm)
                    self.local_grid.step_generation()
                    compute_time = time.time() - t_calc_start
                    local_payload = self.local_grid.matrix[:, 1:-1].flatten()
                    cols_count = self.local_grid.local_w
                else:
                    compute_time, local_payload, cols_count = 0, np.array([], dtype=np.uint8), 0

                # === ETAPE 2 : Rassemblement Gatherv ===
                array_of_cols = np.zeros(self.total_processes, dtype=int) if self.process_rank == 0 else None
                self.global_comm.Gather(np.array([cols_count], dtype=int), array_of_cols, root=0)

                ny = self.universe_dim[0]
                if self.process_rank == 0:
                    counts = array_of_cols * ny
                    displs = np.zeros(self.total_processes, dtype=int)
                    for idx in range(1, self.total_processes):
                        displs[idx] = displs[idx-1] + counts[idx-1]
                    global_flat = np.zeros(displs[-1] + counts[-1], dtype=np.uint8)
                else:
                    counts, displs, global_flat = None, None, None

                self.global_comm.Gatherv(local_payload, [global_flat, counts, displs, MPI.UINT8_T], root=0)
                if self.process_rank == 0: compute_time = time.time() - t_master_wait

            # === ETAPE 3 : Rendu Graphique (Rank 0) ===
            if self.is_standalone:
                t_render_start = time.time()
                self.renderer.render_frame(self.local_grid.matrix[:, 1:-1])
                render_time = time.time() - t_render_start
                self._handle_events()
                print(f"Sync-Col | Calc: {compute_time:.2e}s | Rendu: {render_time:.2e}s\r", end='')
            else:
                if self.process_rank == 0:
                    ny = self.universe_dim[0]
                    final_grid = np.zeros(self.universe_dim, dtype=np.uint8)
                    for k in range(1, self.total_processes):
                        c_start = displs[k] // ny
                        c_end = c_start + array_of_cols[k]
                        final_grid[:, c_start:c_end] = global_flat[displs[k]:displs[k]+counts[k]].reshape((ny, array_of_cols[k]))

                    t_render_start = time.time()
                    self.renderer.render_frame(final_grid)
                    render_time = time.time() - t_render_start
                    self._handle_events()
                    print(f"Workers-Col: {compute_time:.2e}s | Master: {render_time:.2e}s\r", end='')

            self.is_running = self.global_comm.bcast(self.is_running, root=0)

        if self.process_rank == 0: pg.quit()

    def _handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT: self.is_running = False


if __name__ == '__main__':
    sim_app = SimulationController()
    sim_app.execute_simulation()