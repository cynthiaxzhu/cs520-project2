from collections import deque

from utils import get_open_neighbors


def breadth_first_search(spaceship, source):
    queue = deque()
    dist_source = {}
    
    queue.append(source)
    dist_source[source] = 0
    while queue:
        pair = queue.popleft()
        neighbors = get_open_neighbors(pair, spaceship)
        for neighbor in neighbors:
            if neighbor not in dist_source:
                queue.append(neighbor)
                dist_source[neighbor] = dist_source[pair] + 1
    
    return dist_source
