import random


def generate_random_three_hop_path(table_num, par_tabs, foreign_keys):
    start_table = random.choice(range(table_num))
    generated_tables = next(generate_three_hop_path_from_seed(start_table, par_tabs, foreign_keys))
    return generated_tables


def generate_three_hop_path_from_seed(start_table, par_tabs, foreign_keys):
    one_hop_neighbors = set()
    for f, p in foreign_keys:
        if par_tabs[f] == start_table:
            one_hop_neighbors.add(par_tabs[p])
        if par_tabs[p] == start_table:
            one_hop_neighbors.add(par_tabs[f])
    if start_table in one_hop_neighbors:
        one_hop_neighbors.remove(start_table)
    one_hop_neighbors = list(one_hop_neighbors)
    random.shuffle(one_hop_neighbors)

    three_available = False
    for one_neighbor in one_hop_neighbors:
        two_hop_neighbors = set()
        for f, p in foreign_keys:
            if par_tabs[f] == one_neighbor:
                two_hop_neighbors.add(par_tabs[p])
            if par_tabs[p] == one_neighbor:
                two_hop_neighbors.add(par_tabs[f])
        if one_neighbor in two_hop_neighbors:
            two_hop_neighbors.remove(one_neighbor)
        if start_table in two_hop_neighbors:
            two_hop_neighbors.remove(start_table)
        two_hop_neighbors = list(two_hop_neighbors)
        random.shuffle(two_hop_neighbors)
        for two_neighbor in two_hop_neighbors:
            three_available = True
            yield [start_table, one_neighbor, two_neighbor]

    if three_available:
        return
    middle_three_available = False
    for one_neighbor in one_hop_neighbors:
        for two_neighbor in one_hop_neighbors:
            if one_neighbor == two_neighbor:
                continue
            middle_three_available = True
            yield [one_neighbor, start_table, two_neighbor]

    if middle_three_available:
        return
    two_available = False
    for one_neighbor in one_hop_neighbors:
        two_available = True
        yield [start_table, one_neighbor]

    if two_available:
        return
    yield [start_table]

