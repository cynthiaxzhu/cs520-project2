from random import randrange, shuffle

from utils import get_open_neighbors


def place_bot(spaceship):
    bot = None
    while not bot:
        x = randrange(0, spaceship.grid_size)
        y = randrange(0, spaceship.grid_size)
        if spaceship.grid[x][y]:
            bot = (x, y)
    return bot


def place_crew_members(spaceship, bot, num_crew_members):
    crew_members = []
    while len(crew_members) < num_crew_members:
        x = randrange(0, spaceship.grid_size)
        y = randrange(0, spaceship.grid_size)
        new_crew_member = (x, y)
        if (spaceship.grid[x][y] and
            new_crew_member not in crew_members and
            new_crew_member != bot):
            crew_members.append(new_crew_member)
    return crew_members


def place_aliens(spaceship, bot, num_aliens, apothem):
    x_bot, y_bot = bot
    aliens = []
    while len(aliens) < num_aliens:
        x = randrange(0, spaceship.grid_size)
        y = randrange(0, spaceship.grid_size)
        new_alien = (x, y)
        if ((abs(x - x_bot) > apothem or abs(y - y_bot) > apothem) and
            spaceship.grid[x][y] and
            # new_alien not in aliens and  # if aliens must be in different cells
            new_alien != bot):
            aliens.append(new_alien)
    return aliens


def place_all(spaceship, num_crew_members, num_aliens, apothem):
    bot = place_bot(spaceship)
    crew_members = place_crew_members(spaceship, bot, num_crew_members)
    aliens = place_aliens(spaceship, bot, num_aliens, apothem)
    return bot, crew_members, aliens


def move_aliens(spaceship, aliens):
    num_aliens = len(aliens)
    
    shuffle(aliens)
    
    for i in range(0, num_aliens):
        next = get_open_neighbors(aliens[i], spaceship)
        # next.append(aliens[i])  # if alien can stay in place
        rand_num = randrange(0, len(next))
        aliens[i] = next[rand_num]
