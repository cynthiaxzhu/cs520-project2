def index_in_bounds(i, grid_size):
    return -1 < i < grid_size


def get_neighbors(pair, grid_size):
    x, y = pair
    
    possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    
    neighbors = [
        (x_neighbor, y_neighbor)
        for x_neighbor, y_neighbor in possible_neighbors
        if (index_in_bounds(x_neighbor, grid_size) and
            index_in_bounds(y_neighbor, grid_size))
    ]
    
    return neighbors


def get_open_neighbors(pair, spaceship):
    neighbors = get_neighbors(pair, spaceship.grid_size)
    
    open_neighbors = [
        (x_neighbor, y_neighbor)
        for x_neighbor, y_neighbor in neighbors
        if spaceship.grid[x_neighbor][y_neighbor]
    ]
    
    return open_neighbors


def get_closed_neighbors(pair, spaceship):
    neighbors = get_neighbors(pair, spaceship.grid_size)
    
    closed_neighbors = [
        (x_neighbor, y_neighbor)
        for x_neighbor, y_neighbor in neighbors
        if not spaceship.grid[x_neighbor][y_neighbor]
    ]
    
    return closed_neighbors


def get_num_open_neighbors(pair, spaceship):
    return len(get_open_neighbors(pair, spaceship))
