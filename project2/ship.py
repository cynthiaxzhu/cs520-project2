import numpy as np
from itertools import combinations_with_replacement, product
from random import randrange

import utils
from search import breadth_first_search


class Spaceship:
    def __init__(self, grid_size, num_aliens = 1):
        self.grid_size = grid_size
        
        # True if open, False if closed
        self.grid = np.full((self.grid_size, self.grid_size), False)
        
        x = randrange(1, self.grid_size - 1)
        y = randrange(1, self.grid_size - 1)
        self.grid[x][y] = True
        
        closed_cells = self.get_closed_cells_with_one_open_neighbor()
        while closed_cells:
            rand_num = randrange(0, len(closed_cells))
            x, y = closed_cells[rand_num]
            self.grid[x][y] = True
            closed_cells = self.get_closed_cells_with_one_open_neighbor()
        
        open_cells = self.get_open_cells_with_one_open_neighbor()
        for pair in open_cells:
            if randrange(0, 2):
                self.open_one_closed_neighbor(pair)
        
        self.num_open_cells = np.count_nonzero(self.grid)
        
        self.grid_neighbors = [
            [[] for _ in range(self.grid_size)] for _ in range(self.grid_size)
        ]
        self.grid_prob_alien_move_neighbor = (
            np.zeros((self.grid_size, self.grid_size))
        )
        self.init_neighbor_arrays()
        
        self.pairs_neighboring_pairs = None
        if num_aliens == 2:
            self.pairs_neighboring_pairs = self.init_pairs_neighboring_pairs()
        
        self.distances = self.init_distances()
    
    def get_closed_cells_with_one_open_neighbor(self):
        closed_cells = []
        
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if not self.grid[i][j]:
                    if utils.get_num_open_neighbors((i, j), self) == 1:
                        closed_cells.append((i, j))
        
        return closed_cells
    
    def get_open_cells_with_one_open_neighbor(self):
        open_cells = []
        
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.grid[i][j]:
                    if utils.get_num_open_neighbors((i, j), self) == 1:
                        open_cells.append((i, j))
        
        return open_cells
    
    def open_one_closed_neighbor(self, pair):
        closed_neighbors = utils.get_closed_neighbors(pair, self)
        
        if closed_neighbors:
            rand_num = randrange(0, len(closed_neighbors))
            x_neighbor, y_neighbor = closed_neighbors[rand_num]
            self.grid[x_neighbor][y_neighbor] = True
    
    def init_neighbor_arrays(self):
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.grid[i][j]:
                    open_neighbors = utils.get_open_neighbors((i, j), self)
                    self.grid_neighbors[i][j] = open_neighbors
                    self.grid_prob_alien_move_neighbor[i][j] = (
                        1 / len(open_neighbors)
                    )
    
    def init_pairs_neighboring_pairs(self):
        # get open cells
        open_cells = np.argwhere(self.grid).tolist()  # list of lists
        
        # get pairs of open cells
        pairs = (
            list(combinations_with_replacement(open_cells, 2))
        )  # list of tuples of lists
        
        # get neighboring pairs of pairs of open cells
        pairs_neighboring_pairs = {}
        for cell_1, cell_2 in pairs:
            x1, y1 = cell_1
            x2, y2 = cell_2
            neighbors_1 = self.grid_neighbors[x1][y1]  # list of tuples
            neighbors_2 = self.grid_neighbors[x2][y2]  # list of tuples
            pairs_neighboring_pairs[((x1, y1), (x2, y2))] = (
                list(combinations_with_replacement(neighbors_1, 2))
                if cell_1 == cell_2
                else
                list(product(neighbors_1, neighbors_2))
            )  # list of tuples of tuples
        
        return pairs_neighboring_pairs
    
    def init_distances(self):
        distances = {}
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.grid[i][j]:
                    distances[(i, j)] = breadth_first_search(self, (i, j))
        return distances
