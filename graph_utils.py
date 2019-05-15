import random
from copy import deepcopy
from collections import OrderedDict
from hyperparameters import H_PARAM


def graph_checker(graph1, graph2, foreign_keys, primary_keys):
    for t in graph2:
        if int(t) not in graph1:
            return False
    ok_cols = []
    for t in graph1:
        if str(t) not in graph2:
            for col in graph1[t]:
                if col not in primary_keys:
                    return False
            ok_cols += graph1[t]
            continue
    for t in graph1:
        if str(t) not in graph2:
            continue
        t_list = graph1[t]
        graph2_t_list = graph2[str(t)]
        for col in graph2_t_list:
            if col not in t_list:
                return False
        for col in t_list:
            if col not in graph2_t_list:
                ok = False
                for f, p in foreign_keys:
                    if f == col and p in ok_cols:
                        ok = True
                        break
                    if p == col and f in ok_cols:
                        ok = True
                        break
                if not ok:
                    return False
    return True


def graph_checker_origin(graph1, str_graph, foreign_keys, primary_keys):
    if len(graph1) != len(str_graph):
        return False
    for t in graph1:
        if str(t) not in str_graph:
            return False
        t_list = graph1[t]
        t_list.sort()
        graph2_t_list = str_graph[str(t)]
        graph2_t_list.sort()
        if t_list != graph2_t_list:
            return False
    return True


def str_graph_to_num_graph(graph):
    newgraph = {}
    for t in graph:
        newgraph[int(t)] = graph[t]
    return newgraph


def generate_four_hop_path_from_seed(start_table, par_tabs, foreign_keys):
    table_graph = OrderedDict()
    table_graph[start_table] = []
    yield deepcopy(table_graph)

    def next_neighbors(current_table_graph):
        def col_in_graph(col):
            for _, l in current_table_graph.items():
                if col in l:
                    return True
            return False
        current_tables = list(current_table_graph.keys())
        start_table = current_tables[-1]
        one_hop_neighbors = []
        for f, p in foreign_keys:
            # if col_in_graph(f) or col_in_graph(p):
            #     continue
            if par_tabs[f] == start_table and par_tabs[p] not in current_table_graph:
                one_hop_neighbors.append((start_table, f, p, par_tabs[p]))
            if par_tabs[p] == start_table and par_tabs[f] not in current_table_graph:
                one_hop_neighbors.append((start_table, p, f, par_tabs[f]))
        return one_hop_neighbors

    one_neighbors = next_neighbors(current_table_graph=table_graph)
    for start_table, f, p, tab in one_neighbors:
        table_graph[start_table].append((f, p, tab))
        table_graph[tab] = [(p, f, start_table)]
        yield deepcopy(table_graph)
        two_neighbors = next_neighbors(table_graph)
        for start_table2, f2, p2, tab2 in two_neighbors:
            table_graph[start_table2].append((f2, p2, tab2))
            table_graph[tab2] = [(p2, f2, start_table2)]
            yield deepcopy(table_graph)
            three_neighbors = next_neighbors(table_graph)
            for start_table3, f3, p3, tab3 in three_neighbors:
                table_graph[start_table3].append((f3, p3, tab3))
                table_graph[tab3] = [(p3, f3, start_table3)]
                yield deepcopy(table_graph)
                table_graph.pop(tab3)
                table_graph[start_table3].pop()
            table_graph.pop(tab2)
            table_graph[start_table2].pop()
        table_graph.pop(tab)
        table_graph[start_table].pop()


def generate_random_graph_similar_to_label(table_num, par_tabs, foreign_keys, label):
    graph = OrderedDict()
    for t in label:
        graph[int(t)] = label[t]
    appended_num = 0
    # while random.randint(0, 100) < 70:
    if len(label) == 1 or random.randint(0, 100) < 50:
        if appended_num == 2:
            return graph
        graph = append_one_tab(par_tabs, foreign_keys, graph)
        appended_num += 1
    else:
        graph = del_one_tab(par_tabs, foreign_keys, graph)
    return graph


def del_one_tab(par_tabs, foreign_keys, graph):
    for i in range(100):
        delete_tab = random.choice(list(graph.keys()))
        if len(graph[delete_tab]) == 1:
            break
    if len(graph[delete_tab]) != 1:
        return graph
    [col] = graph[delete_tab]
    for f, p in foreign_keys:
        if col == f and par_tabs[p] in graph and p in graph[par_tabs[p]]:
            graph.pop(delete_tab)
            graph[par_tabs[p]].remove(p)
            break
        elif col == p and par_tabs[f] in graph and f in graph[par_tabs[f]]:
            graph.pop(delete_tab)
            graph[par_tabs[f]].remove(f)
            break
    return graph


def append_one_tab(par_tabs, foreign_keys, graph):
    if len(graph) == max(par_tabs):
        return graph
    for i in range(0, 100):
        add_tab = random.choice(range(max(par_tabs)))
        if add_tab in graph:
            continue
        for f, p in foreign_keys:
            if par_tabs[f] == add_tab and par_tabs[p] in graph:
                graph[add_tab] = [f]
                graph[par_tabs[p]].append(p)
                return graph
            elif par_tabs[p] == add_tab and par_tabs[f] in graph:
                graph[add_tab] = [p]
                graph[par_tabs[f]].append(f)
                return graph
    return graph



def generate_random_graph_generate(table_num, par_tabs, foreign_keys):
    percentage = random.randint(0, 100)
    start_table = random.choice(range(table_num))
    if percentage < 33:
        return generate_one_hop_path_from_seed(start_table)
    elif percentage < 66:
        return generate_two_hop_path_from_seed(start_table, par_tabs, foreign_keys)
    else:
        return generate_three_hop_path_from_seed(start_table, par_tabs, foreign_keys)


def generate_three_hop_path_from_seed(start_table, par_tabs, foreign_keys):
    one_hop_neighbors = []
    for f, p in foreign_keys:
        if par_tabs[f] == start_table and par_tabs[p] != start_table:
            one_hop_neighbors.append((f, p, par_tabs[p]))
        if par_tabs[p] == start_table and par_tabs[f] != start_table:
            one_hop_neighbors.append((p, f, par_tabs[f]))
    random.shuffle(one_hop_neighbors)

    copied_foreign_keys = deepcopy(foreign_keys)
    random.shuffle(copied_foreign_keys)

    for f, p, one_neighbor in one_hop_neighbors:
        for f1, p1 in copied_foreign_keys:
            if par_tabs[f1] == one_neighbor and par_tabs[p1] != one_neighbor and par_tabs[p1] != start_table:
                table_graph = OrderedDict()
                table_graph[start_table] = [f]
                table_graph[one_neighbor] = [p, f1]
                table_graph[par_tabs[p1]] = [p1]
                return table_graph
            if par_tabs[p1] == one_neighbor and par_tabs[f1] != one_neighbor and par_tabs[f1] != start_table:
                table_graph = OrderedDict()
                table_graph[start_table] = [f]
                table_graph[one_neighbor] = [p, p1]
                table_graph[par_tabs[f1]] = [f1]
                return table_graph
    return generate_two_hop_path_from_seed(start_table, par_tabs, foreign_keys)


def generate_two_hop_path_from_seed(start_table, par_tabs, foreign_keys):
    copied_foreign_keys = deepcopy(foreign_keys)
    random.shuffle(copied_foreign_keys)
    for f, p in copied_foreign_keys:
        if par_tabs[f] == start_table and par_tabs[p] != start_table:
            table_graph = OrderedDict()
            table_graph[start_table] = [f]
            table_graph[par_tabs[p]] = [p]
            return table_graph
        if par_tabs[p] == start_table and par_tabs[f] != start_table:
            table_graph = OrderedDict()
            table_graph[start_table] = [p]
            table_graph[par_tabs[f]] = [f]
            return table_graph
    return generate_one_hop_path_from_seed(start_table)


def generate_one_hop_path_from_seed(start_table):
    table_graph = OrderedDict()
    table_graph[start_table] = []
    return table_graph


def make_compound_table(table_dict, my_db_id, db_ids):
    if H_PARAM["dev_db_compound_num"] == 0:
        return table_dict[my_db_id]
    selected_db_ids = random.sample(db_ids, H_PARAM["dev_db_compound_num"])
    if my_db_id in selected_db_ids:
        selected_db_ids.remove(my_db_id)

    compound_table = deepcopy(table_dict[my_db_id])
    for dev_db_id in selected_db_ids:
        new_table = table_dict[dev_db_id]
        if random.randint(0, 10) < 5:
            new_table = compound_table
            compound_table = deepcopy(table_dict[dev_db_id])
        compound_table = append_table(compound_table, new_table)
    return compound_table


def append_table(compound_table, new_table):
    for table_name in new_table["table_names"]:
        if table_name in compound_table["table_names"]:
            return compound_table
    new_table_offset = len(compound_table["table_names"])
    new_column_offset = len(compound_table["column_names"]) - 1
    compound_table["table_names"].extend(new_table["table_names"])
    compound_table["table_names_original"].extend(new_table["table_names_original"])
    for p in new_table["primary_keys"]:
        compound_table["primary_keys"].append(p + new_column_offset)
    for f, p in new_table["foreign_keys"]:
        compound_table["foreign_keys"].append([f + new_column_offset, p + new_column_offset])
    compound_table["column_types"].extend(new_table["column_types"])
    for t, name in new_table["column_names_original"][1:]:
        compound_table["column_names_original"].append([t + new_table_offset, name])
    for t, name in new_table["column_names"][1:]:
        compound_table["column_names"].append([t + new_table_offset, name])
    return compound_table



