"""
Simulateur d'Automate Cellulaire (Conway's Game of Life)
Parallélisation par lignes avec MPI (Non-blocking communications)
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

        # Définition de la grille visuelle si les cellules sont assez grandes
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
        # Préparation de la matrice RGB (y, x transposé en x, y pour pygame)
        rgb_matrix = np.zeros((self.universe_shape[1], self.universe_shape[0], 3), dtype=np.uint8)
        grid_t = full_grid_state.T
        
        # Colorisation vectorisée
        rgb_matrix[grid_t == 0] = self.c_empty[:3] 
        rgb_matrix[grid_t == 1] = self.c_alive[:3]

        # Rendu surfarray
        fast_surface = pg.surfarray.make_surface(rgb_matrix)
        fast_surface = pg.transform.flip(fast_surface, False, True)
        fast_surface = pg.transform.scale(fast_surface, (self.total_w, self.total_h))
        
        self.display_surface.blit(fast_surface, (0, 0))
        
        if self.grid_lines_surface:
            self.display_surface.blit(self.grid_lines_surface, (0, 0))
            
        pg.display.update()


class RegionSubGrid:
    """
    Sous-grille gérant un sous-ensemble de l'univers pour un processus Worker.
    """
    def __init__(self, mpi_rank, total_workers, shape, seed_coords=None):
        self.my_rank = mpi_rank
        global_h, self.cols = shape
        
        # Partitionnement de la charge de travail (Row block distribution)
        chunk_size = global_h // total_workers
        remainder = global_h % total_workers
        self.local_h = chunk_size + (1 if mpi_rank < remainder else 0)
        self.offset_y = mpi_rank * chunk_size + min(mpi_rank, remainder)
        
        # Allocation mémoire (+2 pour les halos de communication haut et bas)
        self.matrix = np.zeros((self.local_h + 2, self.cols), dtype=np.uint8)

        # Peuplement initial de la matrice
        if seed_coords:
            for (y_glb, x_glb) in seed_coords:
                if self.offset_y <= y_glb < self.offset_y + self.local_h:
                    y_loc = y_glb - self.offset_y + 1 
                    self.matrix[y_loc, x_glb] = 1
        else:
            self.matrix[1:self.local_h+1, :] = np.random.randint(2, size=(self.local_h, self.cols), dtype=np.uint8)

    def communicate_boundaries(self, comm):
        """
        Synchronisation des halos via communications non-bloquantes (MPI.Request.Waitall).
        Remplace la structure if/else complexe de l'ancien code.
        """
        cluster_size = comm.Get_size()
        my_id = comm.Get_rank()

        if cluster_size == 1:
            self.matrix[0, :] = self.matrix[-2, :]
            self.matrix[-1, :] = self.matrix[1, :]
            return

        # Identification des voisins toriques
        neighbor_up = (my_id - 1) % cluster_size
        neighbor_down = (my_id + 1) % cluster_size

        # Buffers de réception
        buffer_recv_up = np.empty(self.cols, dtype=np.uint8)
        buffer_recv_down = np.empty(self.cols, dtype=np.uint8)

        # Données à envoyer (copie explicite requise pour l'asynchronisme)
        data_send_up = self.matrix[1, :].copy()
        data_send_down = self.matrix[-2, :].copy()

        # Phase de communication non-bloquante (Isend / Irecv)
        mpi_requests = [
            comm.Irecv(buffer_recv_up, source=neighbor_up),
            comm.Irecv(buffer_recv_down, source=neighbor_down),
            comm.Isend(data_send_up, dest=neighbor_up),
            comm.Isend(data_send_down, dest=neighbor_down)
        ]
        
        # Attente de la complétion de tous les échanges
        MPI.Request.Waitall(mpi_requests)
        
        # Mise à jour des lignes fantômes
        self.matrix[0, :] = buffer_recv_up
        self.matrix[-1, :] = buffer_recv_down

    def step_generation(self):
        """
        Application des règles de Conway en utilisant des boucles imbriquées.
        """
        active_rows = self.local_h
        width = self.cols
        future_matrix = np.zeros_like(self.matrix)

        # Parcours de l'espace local
        for row_idx in range(1, active_rows + 1):
            r_top = row_idx - 1
            r_bottom = row_idx + 1
            
            for col_idx in range(width):
                c_left = (col_idx - 1 + width) % width
                c_right = (col_idx + 1) % width
                
                # Comptage manuel des voisins
                alive_neighbors = (
                    self.matrix[r_top, c_left] + self.matrix[r_top, col_idx] + self.matrix[r_top, c_right] +
                    self.matrix[row_idx, c_left] +                             self.matrix[row_idx, c_right] +
                    self.matrix[r_bottom, c_left] + self.matrix[r_bottom, col_idx] + self.matrix[r_bottom, c_right]
                )

                # Application de la logique de survie/naissance
                cell_state = self.matrix[row_idx, col_idx]
                if cell_state == 1: 
                    if alive_neighbors == 2 or alive_neighbors == 3:
                        future_matrix[row_idx, col_idx] = 1 
                    else:
                        future_matrix[row_idx, col_idx] = 0
                else:
                    if alive_neighbors == 3: 
                        future_matrix[row_idx, col_idx] = 1         
                    else:
                        future_matrix[row_idx, col_idx] = 0         
                    
        self.matrix = future_matrix


class SimulationController:
    """
    Contrôleur principal encapsulant la boucle de simulation et la logique de distribution.
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
        
        # Split du communicateur (Affichage vs Calcul)
        split_color = 0 if self.process_rank == 0 else 1
        self.workers_comm = self.global_comm.Split(split_color, self.process_rank)
        if self.process_rank == 0:
            self.workers_comm = None

        self._parse_args_and_setup()
        self.is_running = True

    def _parse_args_and_setup(self):
        target_pattern = 'glider' if len(sys.argv) <= 1 else sys.argv[1]
        res_width, res_height = 800, 800
        if len(sys.argv) > 3:
            res_width, res_height = int(sys.argv[2]), int(sys.argv[3])

        if self.process_rank == 0:
            pg.init()
            print(f"[*] Configuration: Pattern={target_pattern}, Resolution={res_width}x{res_height}")

        if target_pattern not in self.AVAILABLE_SEEDS:
            if self.process_rank == 0:
                print("Erreur: Pattern introuvable. Disponibles:", list(self.AVAILABLE_SEEDS.keys()))
            sys.exit(1)

        self.universe_dim, initial_coords = self.AVAILABLE_SEEDS[target_pattern]
        self.is_standalone = (self.total_processes == 1)

        # Initialisation composants
        self.local_grid = None
        if self.is_standalone:
            self.local_grid = RegionSubGrid(0, 1, self.universe_dim, initial_coords)
        elif self.process_rank != 0:
            self.local_grid = RegionSubGrid(self.process_rank - 1, self.total_processes - 1, self.universe_dim, initial_coords)

        self.renderer = PygameRenderer((res_width, res_height), self.universe_dim) if self.process_rank == 0 else None

    def execute_simulation(self):
        while self.is_running:
            # === ETAPE 1 : Calcul Local ===
            if self.is_standalone:
                t0_calc = time.time()
                self.local_grid.communicate_boundaries(self.global_comm)
                self.local_grid.step_generation()
                compute_time = time.time() - t0_calc
                
                payload_data = None
                rows_count = self.local_grid.local_h
            else:
                if self.process_rank == 0:
                    t_master_wait = time.time()
                
                if self.process_rank != 0:
                    t0_calc = time.time()
                    self.local_grid.communicate_boundaries(self.workers_comm)
                    self.local_grid.step_generation()
                    compute_time = time.time() - t0_calc

                    payload_data = self.local_grid.matrix[1:-1, :].flatten()
                    rows_count = self.local_grid.local_h
                else:
                    compute_time = 0
                    payload_data = np.array([], dtype=np.uint8)
                    rows_count = 0

                # === ETAPE 2 : Assemblage (Gatherv) ===
                array_of_rows = np.zeros(self.total_processes, dtype=int) if self.process_rank == 0 else None
                self.global_comm.Gather(np.array([rows_count], dtype=int), array_of_rows, root=0)

                w_cols = self.universe_dim[1]
                if self.process_rank == 0:
                    counts_per_proc = array_of_rows * w_cols
                    offsets = np.zeros(self.total_processes, dtype=int)
                    for idx in range(1, self.total_processes):
                        offsets[idx] = offsets[idx-1] + counts_per_proc[idx-1]
                    
                    total_buffer_size = offsets[-1] + counts_per_proc[-1]
                    assembled_flat_grid = np.zeros(total_buffer_size, dtype=np.uint8)
                else:
                    counts_per_proc, offsets, assembled_flat_grid = None, None, None

                self.global_comm.Gatherv(payload_data, [assembled_flat_grid, counts_per_proc, offsets, MPI.UINT8_T], root=0)
                
                if self.process_rank == 0:
                    compute_time = time.time() - t_master_wait

            # === ETAPE 3 : Rendu Graphique (Rank 0) ===
            if self.is_standalone:
                t0_render = time.time()
                self.renderer.render_frame(self.local_grid.matrix[1:-1, :])
                render_time = time.time() - t0_render
                self._handle_events()
                print(f"Sync | Calcul: {compute_time:.2e}s | Rendu: {render_time:.2e}s\r", end='')
                
            else:
                if self.process_rank == 0:
                    w_cols = self.universe_dim[1]
                    final_matrix2d = np.zeros(self.universe_dim, dtype=np.uint8)
                    for k in range(1, self.total_processes):
                        row_start = offsets[k] // w_cols
                        row_end = row_start + array_of_rows[k]
                        segment = assembled_flat_grid[offsets[k]:offsets[k]+counts_per_proc[k]]
                        final_matrix2d[row_start:row_end, :] = segment.reshape((array_of_rows[k], w_cols))

                    t0_render = time.time()
                    self.renderer.render_frame(final_matrix2d)
                    render_time = time.time() - t0_render

                    self._handle_events()
                    print(f"Workers: {compute_time:.2e}s | Maître(Rendu): {render_time:.2e}s\r", end='')

            # === ETAPE 4 : Synchronisation d'état ===
            self.is_running = self.global_comm.bcast(self.is_running, root=0)

        if self.process_rank == 0:
            pg.quit()

    def _handle_events(self):
        """Gestion des inputs clavier/fenêtre"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False


if __name__ == '__main__':
    sim_app = SimulationController()
    sim_app.execute_simulation()