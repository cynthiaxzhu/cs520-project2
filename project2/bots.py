import numpy as np
from random import random, randrange
from scipy.special import binom

import actions
from utils import get_open_neighbors


def get_detection_square_slices(spaceship, bot, apothem):
    x_bot, y_bot = bot
    
    up = max(x_bot - apothem, 0)
    down = min(x_bot + apothem, spaceship.grid_size - 1)
    left = max(y_bot - apothem, 0)
    right = min(y_bot + apothem, spaceship.grid_size - 1)
    
    slice_x = slice(up, down + 1)
    slice_y = slice(left, right + 1)
    
    return slice_x, slice_y


def get_pairs_with_replacement(cond):
    return np.multiply.outer(cond, cond)


def condition_inside_detection_square(spaceship, bot, apothem):
    slice_x, slice_y = get_detection_square_slices(spaceship, bot, apothem)
    
    cond = np.full((spaceship.grid_size, spaceship.grid_size), False)
    cond[slice_x, slice_y] = spaceship.grid[slice_x, slice_y]
    
    return cond


def condition_outside_detection_square(spaceship, bot, apothem):
    slice_x, slice_y = get_detection_square_slices(spaceship, bot, apothem)
    
    cond = spaceship.grid.copy()
    cond[slice_x, slice_y] = False
    
    return cond


def condition_pairs_without_replacement(spaceship, bot):
    grid_size = spaceship.grid_size
    x_bot, y_bot = bot
    
    # pairs without replacement
    cond = np.outer(spaceship.grid.reshape(-1), spaceship.grid.reshape(-1))
    np.fill_diagonal(cond, False)  # no replacement
    cond.shape = (grid_size, grid_size, grid_size, grid_size)
    
    # exclude pairs with bot
    cond[x_bot, y_bot, :, :] = False
    cond[:, :, x_bot, y_bot] = False
    
    return cond


def condition_pairs_with_replacement(spaceship, bot):
    x_bot, y_bot = bot
    
    # pairs with replacement
    cond = get_pairs_with_replacement(spaceship.grid)
    
    # exclude pairs with bot
    cond[x_bot, y_bot, :, :] = False
    cond[:, :, x_bot, y_bot] = False
    
    return cond


def condition_pairs_with_replacement_inside_detection_square(
    spaceship, bot, apothem
):
    cond_pairs = condition_pairs_with_replacement(spaceship, bot)
    cond_pairs_outside = (
        condition_pairs_with_replacement_outside_detection_square(
            spaceship, bot, apothem
        )
    )
    cond_pairs_inside = np.logical_xor(cond_pairs, cond_pairs_outside)
    return cond_pairs_inside


def condition_pairs_with_replacement_outside_detection_square(
    spaceship, bot, apothem
):
    cond_outside = condition_outside_detection_square(spaceship, bot, apothem)
    cond_pairs_outside = get_pairs_with_replacement(cond_outside)
    return cond_pairs_outside


def init_prob_alien_1(spaceship, bot, apothem):
    cond_outside = condition_outside_detection_square(spaceship, bot, apothem)
    
    p = 1 / np.count_nonzero(cond_outside)
    
    prob_alien_1 = np.where(cond_outside, p, 0)
    
    return prob_alien_1


def update_prob_alien_1_not_found(prob_alien_1, bot):
    x_bot, y_bot = bot
    
    p_alien_in_bot = prob_alien_1[x_bot, y_bot]
    p_alien_not_in_bot = 1 - p_alien_in_bot
    
    prob_alien_1[x_bot, y_bot] = 0
    prob_alien_1 /= p_alien_not_in_bot


def update_prob_alien_1_detected(prob_alien_1, spaceship, bot, apothem):
    cond_outside = condition_outside_detection_square(spaceship, bot, apothem)
    prob_alien_1[cond_outside] = 0
    
    """
    added if statement to fix error when running Bot 6
        RuntimeWarning: invalid value encountered in divide
        prob_alien_1 /= sum
    """
    if np.any(prob_alien_1):
        sum = np.sum(prob_alien_1)
        prob_alien_1 /= sum
    else:
        x_bot, y_bot = bot
        
        cond_inside = (
            condition_inside_detection_square(spaceship, bot, apothem)
        )
        
        p = 1 / (np.count_nonzero(cond_inside) - 1)
        
        prob_alien_1[cond_inside] = p
        prob_alien_1[x_bot, y_bot] = 0


def update_prob_alien_1_not_detected(prob_alien_1, spaceship, bot, apothem):
    cond_inside = condition_inside_detection_square(spaceship, bot, apothem)
    prob_alien_1[cond_inside] = 0
    
    """
    added if statement to fix error when running Bot 6
        RuntimeWarning: invalid value encountered in divide
        prob_alien_1 /= sum
    """
    if np.any(prob_alien_1):
        sum = np.sum(prob_alien_1)
        prob_alien_1 /= sum
    else:
        cond_outside = (
            condition_outside_detection_square(spaceship, bot, apothem)
        )
        p = 1 / np.count_nonzero(cond_outside)
        prob_alien_1[cond_outside] = p


def update_prob_alien_1_alien_moves(prob_alien_1, spaceship):
    prob = prob_alien_1 * spaceship.grid_prob_alien_move_neighbor
    
    for i in range(0, spaceship.grid_size):
        for j in range(0, spaceship.grid_size):
            if spaceship.grid[i][j]:
                p = 0
                for x_neighbor, y_neighbor in spaceship.grid_neighbors[i][j]:
                    p += prob[x_neighbor, y_neighbor]
                prob_alien_1[i, j] = p


def init_prob_alien_2(spaceship, bot, apothem):
    cond_outside = condition_outside_detection_square(spaceship, bot, apothem)
    
    cond_pairs_outside = get_pairs_with_replacement(cond_outside)
    
    p = 1 / binom(np.count_nonzero(cond_outside) + 1, 2)
    
    prob_alien_2 = np.where(cond_pairs_outside, p, 0)
    for i in range(0, spaceship.grid_size):
        for j in range(0, spaceship.grid_size):
            prob_alien_2[i, j, i, j] *= 2
    
    return prob_alien_2


def update_prob_alien_2_not_found(prob_alien_2, bot):
    x_bot, y_bot = bot
    
    p_alien_in_bot = np.sum(prob_alien_2[x_bot, y_bot])
    p_alien_not_in_bot = 1 - p_alien_in_bot
    
    prob_alien_2[x_bot, y_bot, :, :] = 0
    prob_alien_2[:, :, x_bot, y_bot] = 0
    prob_alien_2 /= p_alien_not_in_bot


def update_prob_alien_2_detected(prob_alien_2, spaceship, bot, apothem):
    cond_pairs_outside = (
        condition_pairs_with_replacement_outside_detection_square(
            spaceship, bot, apothem
        )
    )
    prob_alien_2[cond_pairs_outside] = 0
    
    sum = np.sum(prob_alien_2) / 2  # prob_alien_2 is symmetric
    prob_alien_2 /= sum


def update_prob_alien_2_not_detected(prob_alien_2, spaceship, bot, apothem):
    cond_pairs_inside = (
        condition_pairs_with_replacement_inside_detection_square(
            spaceship, bot, apothem
        )
    )
    prob_alien_2[cond_pairs_inside] = 0
    
    sum = np.sum(prob_alien_2) / 2  # prob_alien_2 is symmetric
    prob_alien_2 /= sum


def update_prob_alien_2_aliens_move(prob_alien_2, spaceship):
    prob = (
        prob_alien_2
        * spaceship.grid_prob_alien_move_neighbor[:, :, np.newaxis, np.newaxis]
        * spaceship.grid_prob_alien_move_neighbor[np.newaxis, np.newaxis, :, :]
    )
    
    for pair, neighboring_pairs in spaceship.pairs_neighboring_pairs.items():
        jx1, jy1 = pair[0]
        jx2, jy2 = pair[1]
        
        p = 0
        for neighboring_pair in neighboring_pairs:
            kx1, ky1 = neighboring_pair[0]
            kx2, ky2 = neighboring_pair[1]
            if neighboring_pair[0] == neighboring_pair[1]:
                p += prob[kx1, ky1, kx2, ky2] / 2
            else:
                p += prob[kx1, ky1, kx2, ky2]
        
        if pair[0] == pair[1]:
            prob_alien_2[jx1, jy1, jx2, jy2] = p * 2
        else:
            prob_alien_2[jx1, jy1, jx2, jy2] = p
            prob_alien_2[jx2, jy2, jx1, jy1] = p
    
    # normalize
    sum = np.sum(prob_alien_2) / 2  # prob_alien_2 is symmetric
    prob_alien_2 /= sum


def init_prob_crew_1(spaceship, bot):
    x_bot, y_bot = bot
    
    p = 1 / (spaceship.num_open_cells - 1)
    
    prob_crew_1 = np.where(spaceship.grid, p, 0)
    prob_crew_1[x_bot, y_bot] = 0
    
    return prob_crew_1


def update_prob_crew_1_not_found(prob_crew_1, spaceship, bot):
    x_bot, y_bot = bot
    
    p_crew_in_bot = prob_crew_1[x_bot, y_bot]
    
    """
    added if statement to fix error when running Bot 3
        RuntimeWarning: divide by zero encountered in divide
        prob_crew_1 /= p_crew_not_in_bot
    """
    if p_crew_in_bot == 1:
        # init_prob_crew_1() in place
        p = 1 / (spaceship.num_open_cells - 1)
        
        prob_crew_1[spaceship.grid] = p
        prob_crew_1[~spaceship.grid] = 0
        prob_crew_1[x_bot, y_bot] = 0
    else:
        p_crew_not_in_bot = 1 - p_crew_in_bot
        
        prob_crew_1[x_bot, y_bot] = 0
        prob_crew_1 /= p_crew_not_in_bot


def update_prob_crew_1_detected(prob_crew_1, spaceship, bot, alpha):
    for i in range(0, spaceship.grid_size):
        for j in range(0, spaceship.grid_size):
            if spaceship.grid[i][j]:
                p = np.exp(-alpha * (spaceship.distances[bot][(i, j)] - 1))
                prob_crew_1[i, j] *= p
    
    sum = np.sum(prob_crew_1)
    prob_crew_1 /= sum


def update_prob_crew_1_not_detected(prob_crew_1, spaceship, bot, alpha):
    x_bot, y_bot = bot
    
    for i in range(0, spaceship.grid_size):
        for j in range(0, spaceship.grid_size):
            if spaceship.grid[i][j]:
                q = 1 - np.exp(-alpha * (spaceship.distances[bot][(i, j)] - 1))
                prob_crew_1[i, j] *= q
    prob_crew_1[x_bot, y_bot] = 0  # replace negative zero with positive zero
    
    sum = np.sum(prob_crew_1)
    prob_crew_1 /= sum


def init_prob_crew_2(spaceship, bot):
    cond = condition_pairs_without_replacement(spaceship, bot)
    
    p = 1 / (
        binom(spaceship.num_open_cells, 2) - (spaceship.num_open_cells - 1)
    )
    
    prob_crew_2 = np.where(cond, p, 0)
    
    return prob_crew_2


def get_prob_crew_1(prob_crew_2, bot):
    x_bot, y_bot = bot
    
    prob_crew_1 = prob_crew_2[x_bot][y_bot].copy()
    sum = np.sum(prob_crew_1)
    prob_crew_1 /= sum
    
    return prob_crew_1


def update_prob_crew_2_not_found(prob_crew_2, bot):
    x_bot, y_bot = bot
    
    p_crew_in_bot = np.sum(prob_crew_2[x_bot, y_bot])
    p_crew_not_in_bot = 1 - p_crew_in_bot
    
    prob_crew_2[x_bot, y_bot, :, :] = 0
    prob_crew_2[:, :, x_bot, y_bot] = 0
    prob_crew_2 /= p_crew_not_in_bot


def update_prob_crew_2_detected(prob_crew_2, spaceship, bot, alpha):
    grid_size = spaceship.grid_size
    
    prob_j1 = np.zeros((grid_size, grid_size, grid_size, grid_size))
    prob_j2 = np.zeros((grid_size, grid_size, grid_size, grid_size))
    
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            if spaceship.grid[i][j]:
                p = np.exp(-alpha * (spaceship.distances[bot][(i, j)] - 1))
                prob_j1[i, j, :, :] = p
                prob_j2[:, :, i, j] = p
    
    prob_crew_2 *= prob_j1 + prob_j2 - prob_j1 * prob_j2
    
    sum = np.sum(prob_crew_2) / 2  # prob_crew_2 is symmetric
    prob_crew_2 /= sum


def update_prob_crew_2_not_detected(prob_crew_2, spaceship, bot, alpha):
    grid_size = spaceship.grid_size
    x_bot, y_bot = bot
    
    prob_not_j1 = np.zeros((grid_size, grid_size, grid_size, grid_size))
    prob_not_j2 = np.zeros((grid_size, grid_size, grid_size, grid_size))
    
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            if spaceship.grid[i][j]:
                q = 1 - np.exp(-alpha * (spaceship.distances[bot][(i, j)] - 1))
                prob_not_j1[i, j, :, :] = q
                prob_not_j2[:, :, i, j] = q
    
    prob_crew_2 *= prob_not_j1 * prob_not_j2
    
    # replace negative zero with positive zero
    prob_crew_2[x_bot, y_bot, :, :] = 0
    prob_crew_2[:, :, x_bot, y_bot] = 0
    
    sum = np.sum(prob_crew_2) / 2  # prob_crew_2 is symmetric
    prob_crew_2 /= sum


def decide_next_bot_1(spaceship, bot, prob_alien, prob_crew):
    next = get_open_neighbors(bot, spaceship)
    next_prob_crew = {}
    for x, y in next:
        if np.sum(prob_alien[x, y]) == 0:
            next_prob_crew[(x, y)] = np.sum(prob_crew[x, y])
    if not next_prob_crew:
        return bot
    max_prob_crew = max(next_prob_crew.values())
    next = [n for n, p in next_prob_crew.items() if p == max_prob_crew]
    rand_num = randrange(0, len(next))
    return next[rand_num]


def decide_next_bot_2(spaceship, bot, prob_alien, prob_crew):
    next = get_open_neighbors(bot, spaceship)
    next_diff = {}
    for x, y in next:
        next_diff[(x, y)] = np.sum(prob_crew[x, y]) - np.sum(prob_alien[x, y])
    max_diff = max(next_diff.values())
    next = [n for n, p in next_diff.items() if p == max_diff]
    rand_num = randrange(0, len(next))
    return next[rand_num]


def decide_next_bot_5(spaceship, bot, aliens, apothem, prob_alien, prob_crew):
    next = get_open_neighbors(bot, spaceship)
    if detect_alien(bot, aliens, apothem):
        next_prob_alien = {}
        for x, y in next:
            next_prob_alien[(x, y)] = np.sum(prob_alien[x, y])
        min_prob_alien = min(next_prob_alien.values())
        next = [n for n, p in next_prob_alien.items() if p == min_prob_alien]
    else:
        next_prob_crew = {}
        for x, y in next:
            next_prob_crew[(x, y)] = np.sum(prob_crew[x, y])
        max_prob_crew = max(next_prob_crew.values())
        next = [n for n, p in next_prob_crew.items() if p == max_prob_crew]
    rand_num = randrange(0, len(next))
    return next[rand_num]


def decide_next_bot_8(spaceship, bot, aliens, apothem, prob_alien, prob_crew):
    next = get_open_neighbors(bot, spaceship)
    if detect_alien(bot, aliens, apothem):
        next_diff = {}
        for x, y in next:
            next_diff[(x, y)] = (
                np.sum(prob_crew[x, y]) - np.sum(prob_alien[x, y])
            )
        max_diff = max(next_diff.values())
        next = [n for n, p in next_diff.items() if p == max_diff]
    else:
        next_prob_crew = {}
        for x, y in next:
            next_prob_crew[(x, y)] = np.sum(prob_crew[x, y])
        max_prob_crew = max(next_prob_crew.values())
        next = [n for n, p in next_prob_crew.items() if p == max_prob_crew]
    rand_num = randrange(0, len(next))
    return next[rand_num]


def detect_alien(bot, aliens, apothem):
    x_bot, y_bot = bot
    for alien in aliens:
        x_alien, y_alien = alien
        if (abs(x_alien - x_bot) <= apothem and
            abs(y_alien - y_bot) <= apothem):
            return True
    return False


def detect_crew_member(bot, crew_members, alpha, distances):
    for crew_member in crew_members:
        p = np.exp(-alpha * (distances[bot][crew_member] - 1))
        if random() < p:
            return True
    return False


def crew_1_alien_1(
    spaceship, apothem, alpha,
    bot, crew_members, aliens,
    prob_alien_1, prob_crew_1,
    num_saved = 0, num_time_steps = 0, is_success = 0,
    bot_num = 1
):
    while crew_members:
        num_time_steps += 1
        
        if bot_num == 2:
            bot = decide_next_bot_2(spaceship, bot, prob_alien_1, prob_crew_1)
        elif bot_num == 5:
            bot = decide_next_bot_5(
                spaceship, bot, aliens, apothem, prob_alien_1, prob_crew_1
            )
        else:
            bot = decide_next_bot_1(spaceship, bot, prob_alien_1, prob_crew_1)
        
        if bot in aliens:
            break
        else:
            update_prob_alien_1_not_found(prob_alien_1, bot)
        
        if bot in crew_members:
            num_saved += 1
            crew_members.remove(bot)
            if not crew_members:
                is_success = 1
                break
        update_prob_crew_1_not_found(prob_crew_1, spaceship, bot)
        
        if detect_alien(bot, aliens, apothem):
            update_prob_alien_1_detected(
                prob_alien_1, spaceship, bot, apothem
            )
        else:
            update_prob_alien_1_not_detected(
                prob_alien_1, spaceship, bot, apothem
            )
        
        if detect_crew_member(bot, crew_members, alpha, spaceship.distances):
            update_prob_crew_1_detected(prob_crew_1, spaceship, bot, alpha)
        else:
            update_prob_crew_1_not_detected(prob_crew_1, spaceship, bot, alpha)
        
        actions.move_aliens(spaceship, aliens)
        if bot in aliens:
            break
        else:
            update_prob_alien_1_alien_moves(prob_alien_1, spaceship)
    
    return num_saved, num_time_steps, is_success


def crew_2_alien_1_found_1(
    spaceship, apothem, alpha,
    bot, crew_members, aliens,
    prob_alien_1, prob_crew_2,
    num_saved, num_time_steps, is_success,
    bot_num = 4
):
    # reduce prob_crew_2 to prob_crew_1
    prob_crew_1 = get_prob_crew_1(prob_crew_2, bot)
    
    # finish current iteration
    if detect_alien(bot, aliens, apothem):
        update_prob_alien_1_detected(prob_alien_1, spaceship, bot, apothem)
    else:
        update_prob_alien_1_not_detected(prob_alien_1, spaceship, bot, apothem)
    
    if detect_crew_member(bot, crew_members, alpha, spaceship.distances):
        update_prob_crew_1_detected(prob_crew_1, spaceship, bot, alpha)
    else:
        update_prob_crew_1_not_detected(prob_crew_1, spaceship, bot, alpha)
    
    actions.move_aliens(spaceship, aliens)
    if bot in aliens:
        return num_saved, num_time_steps, is_success
    else:
        update_prob_alien_1_alien_moves(prob_alien_1, spaceship)
    
    # start next iteration
    return crew_1_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_1,
        num_saved, num_time_steps, is_success,
        bot_num
    )


def crew_2_alien_1(
    spaceship, apothem, alpha,
    bot, crew_members, aliens,
    prob_alien_1, prob_crew_2,
    num_saved = 0, num_time_steps = 0, is_success = 0,
    bot_num = 4
):
    while crew_members:
        num_time_steps += 1
        
        if bot_num == 5:
            bot = decide_next_bot_5(
                spaceship, bot, aliens, apothem, prob_alien_1, prob_crew_2
            )
        else:
            bot = decide_next_bot_1(spaceship, bot, prob_alien_1, prob_crew_2)
        
        if bot in aliens:
            break
        else:
            update_prob_alien_1_not_found(prob_alien_1, bot)
        
        if bot in crew_members:
            num_saved += 1
            crew_members.remove(bot)
            return crew_2_alien_1_found_1(
                spaceship, apothem, alpha,
                bot, crew_members, aliens,
                prob_alien_1, prob_crew_2,
                num_saved, num_time_steps, is_success,
                bot_num
            )
        else:
            update_prob_crew_2_not_found(prob_crew_2, bot)
        
        if detect_alien(bot, aliens, apothem):
            update_prob_alien_1_detected(
                prob_alien_1, spaceship, bot, apothem
            )
        else:
            update_prob_alien_1_not_detected(
                prob_alien_1, spaceship, bot, apothem
            )
        
        if detect_crew_member(bot, crew_members, alpha, spaceship.distances):
            update_prob_crew_2_detected(prob_crew_2, spaceship, bot, alpha)
        else:
            update_prob_crew_2_not_detected(prob_crew_2, spaceship, bot, alpha)
        
        actions.move_aliens(spaceship, aliens)
        if bot in aliens:
            break
        else:
            update_prob_alien_1_alien_moves(prob_alien_1, spaceship)
    
    return num_saved, num_time_steps, is_success


def crew_1_alien_2(
    spaceship, apothem, alpha,
    bot, crew_members, aliens,
    prob_alien_2, prob_crew_1,
    num_saved = 0, num_time_steps = 0, is_success = 0,
    bot_num = 7
):
    while crew_members:
        num_time_steps += 1
        
        if bot_num == 8:
            bot = decide_next_bot_8(
                spaceship, bot, aliens, apothem, prob_alien_2, prob_crew_1
            )
        else:
            bot = decide_next_bot_1(spaceship, bot, prob_alien_2, prob_crew_1)
        
        if bot in aliens:
            break
        else:
            update_prob_alien_2_not_found(prob_alien_2, bot)
        
        if bot in crew_members:
            num_saved += 1
            crew_members.remove(bot)
            if not crew_members:
                is_success = 1
                break
        update_prob_crew_1_not_found(prob_crew_1, spaceship, bot)
        
        if detect_alien(bot, aliens, apothem):
            update_prob_alien_2_detected(
                prob_alien_2, spaceship, bot, apothem
            )
        else:
            update_prob_alien_2_not_detected(
                prob_alien_2, spaceship, bot, apothem
            )
        
        if detect_crew_member(bot, crew_members, alpha, spaceship.distances):
            update_prob_crew_1_detected(prob_crew_1, spaceship, bot, alpha)
        else:
            update_prob_crew_1_not_detected(prob_crew_1, spaceship, bot, alpha)
        
        actions.move_aliens(spaceship, aliens)
        if bot in aliens:
            break
        else:
            update_prob_alien_2_aliens_move(prob_alien_2, spaceship)
    
    return num_saved, num_time_steps, is_success


def crew_2_alien_2_found_1(
    spaceship, apothem, alpha,
    bot, crew_members, aliens,
    prob_alien_2, prob_crew_2,
    num_saved, num_time_steps, is_success,
    bot_num = 7
):
    # reduce prob_crew_2 to prob_crew_1
    prob_crew_1 = get_prob_crew_1(prob_crew_2, bot)
    
    # finish current iteration
    if detect_alien(bot, aliens, apothem):
        update_prob_alien_2_detected(prob_alien_2, spaceship, bot, apothem)
    else:
        update_prob_alien_2_not_detected(prob_alien_2, spaceship, bot, apothem)
    
    if detect_crew_member(bot, crew_members, alpha, spaceship.distances):
        update_prob_crew_1_detected(prob_crew_1, spaceship, bot, alpha)
    else:
        update_prob_crew_1_not_detected(prob_crew_1, spaceship, bot, alpha)
    
    actions.move_aliens(spaceship, aliens)
    if bot in aliens:
        return num_saved, num_time_steps, is_success
    else:
        update_prob_alien_2_aliens_move(prob_alien_2, spaceship)
    
    # start next iteration
    return crew_1_alien_2(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_2, prob_crew_1,
        num_saved, num_time_steps, is_success,
        bot_num
    )


def crew_2_alien_2(
    spaceship, apothem, alpha,
    bot, crew_members, aliens,
    prob_alien_2, prob_crew_2,
    num_saved = 0, num_time_steps = 0, is_success = 0,
    bot_num = 7
):
    while crew_members:
        num_time_steps += 1
        
        if bot_num == 8:
            bot = decide_next_bot_8(
                spaceship, bot, aliens, apothem, prob_alien_2, prob_crew_2
            )
        else:
            bot = decide_next_bot_1(spaceship, bot, prob_alien_2, prob_crew_2)
        
        if bot in aliens:
            break
        else:
            update_prob_alien_2_not_found(prob_alien_2, bot)
        
        if bot in crew_members:
            num_saved += 1
            crew_members.remove(bot)
            return crew_2_alien_2_found_1(
                spaceship, apothem, alpha,
                bot, crew_members, aliens,
                prob_alien_2, prob_crew_2,
                num_saved, num_time_steps, is_success,
                bot_num
            )
        else:
            update_prob_crew_2_not_found(prob_crew_2, bot)
        
        if detect_alien(bot, aliens, apothem):
            update_prob_alien_2_detected(
                prob_alien_2, spaceship, bot, apothem
            )
        else:
            update_prob_alien_2_not_detected(
                prob_alien_2, spaceship, bot, apothem
            )
        
        if detect_crew_member(bot, crew_members, alpha, spaceship.distances):
            update_prob_crew_2_detected(prob_crew_2, spaceship, bot, alpha)
        else:
            update_prob_crew_2_not_detected(prob_crew_2, spaceship, bot, alpha)
        
        actions.move_aliens(spaceship, aliens)
        if bot in aliens:
            break
        else:
            update_prob_alien_2_aliens_move(prob_alien_2, spaceship)
    
    return num_saved, num_time_steps, is_success


def bot_1(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 1,
        num_aliens = 1,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_1 = init_prob_alien_1(spaceship, bot, apothem)
    prob_crew_1 = init_prob_crew_1(spaceship, bot)
    
    return crew_1_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_1
    )


def bot_2(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 1,
        num_aliens = 1,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_1 = init_prob_alien_1(spaceship, bot, apothem)
    prob_crew_1 = init_prob_crew_1(spaceship, bot)
    
    return crew_1_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_1,
        bot_num = 2
    )


def bot_3(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 2,
        num_aliens = 1,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_1 = init_prob_alien_1(spaceship, bot, apothem)
    prob_crew_1 = init_prob_crew_1(spaceship, bot)
    
    return crew_1_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_1
    )


def bot_4(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 2,
        num_aliens = 1,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_1 = init_prob_alien_1(spaceship, bot, apothem)
    prob_crew_2 = init_prob_crew_2(spaceship, bot)
    
    return crew_2_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_2
    )


def bot_5(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 2,
        num_aliens = 1,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_1 = init_prob_alien_1(spaceship, bot, apothem)
    prob_crew_2 = init_prob_crew_2(spaceship, bot)
    
    return crew_2_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_2,
        bot_num = 5
    )


def bot_6(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 2,
        num_aliens = 2,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_1 = init_prob_alien_1(spaceship, bot, apothem)
    prob_crew_1 = init_prob_crew_1(spaceship, bot)
    
    return crew_1_alien_1(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_1, prob_crew_1
    )


def bot_7(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 2,
        num_aliens = 2,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_2 = init_prob_alien_2(spaceship, bot, apothem)
    prob_crew_2 = init_prob_crew_2(spaceship, bot)
    
    return crew_2_alien_2(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_2, prob_crew_2
    )


def bot_8(spaceship, apothem, alpha):
    bot, crew_members, aliens = actions.place_all(
        spaceship = spaceship,
        num_crew_members = 2,
        num_aliens = 2,
        apothem = apothem
    )
    
    num_saved = 0
    num_time_steps = 0
    is_success = 0
    
    prob_alien_2 = init_prob_alien_2(spaceship, bot, apothem)
    prob_crew_2 = init_prob_crew_2(spaceship, bot)
    
    return crew_2_alien_2(
        spaceship, apothem, alpha,
        bot, crew_members, aliens,
        prob_alien_2, prob_crew_2,
        bot_num = 8
    )
