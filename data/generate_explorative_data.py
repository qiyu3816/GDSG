"""
The script is used to generate the exploration data for the multi-server multi-user offloading problem.
The solution is based on the max-flow-min-cost algorithm. Some heuristic functions are used.
"""

import datetime
import os

import yaml
import random
import time

from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph(server_num, user_num):
    G = nx.DiGraph()
    servers = ['s'+str(i) for i in range(server_num)]
    users = ['u'+str(i) for i in range(user_num)]
    G.add_nodes_from(servers, bipartite=0)
    G.add_nodes_from(users, bipartite=1)

    for user in users:
        random_weights = [0.1, 0.75, 0.1, 0.05]
        num_servers = random.choices([1, 2, 3, 4], weights=random_weights, k=1)[0]
        for server in random.sample(servers, num_servers):
            G.add_edge(user, server)

    for server in servers:
        if G.in_degree(server) == 0:
            random_weights = [i / user_num for i in range(user_num)]
            node = random.choices([i for i in range(user_num)], weights=random_weights, k=1)[0]
            G.add_edge(users[node], server)

    return G


def draw_graph(G):
    """
    Simple graph visualization with off-the-shelf nx implementation.
    """
    pos = nx.spring_layout(G)  # node position
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black')
    plt.show()


def generate_weights_randomly(ws, edges, server_num, random_num=60):
    """
    Generate random weights for the edges based on the source heuristic weights.
    """
    for i in range(server_num):
        server_edges = np.where(edges[:, 1] == i)[0]
        total_weight = np.sum(ws[server_edges])
        if total_weight < 1:
            ws[server_edges] += (1 - total_weight) / len(server_edges)
    weights_list = [ws]
    for r in range(random_num):
        new_ws = ws.copy()
        for i in range(server_num):
            server_edges = np.where(edges[:, 1] == i)[0]
            if len(server_edges) > 1:
                change_edge = np.random.choice(server_edges)
                other_edges = server_edges[server_edges != change_edge]

                other_ws_sum = np.sum(ws[other_edges])
                change_value = np.random.uniform(-ws[change_edge] + 0.05, other_ws_sum)
                new_ws[change_edge] += change_value

                adjust_ratio = ws[other_edges] / other_ws_sum
                new_ws[other_edges] -= adjust_ratio * change_value  # ensure the in_dim is not changed and the weights are valid
            else:
                new_ws[server_edges[0]] = 1
        weights_list.append(new_ws)
    return np.array(weights_list)


def solve_once(server_num, user_num, adj_mat, points, edges, local_sat, offload_sat,
               cost_locals, trans_costs, offload_total_costs, least_ratios):
    """
    Use max-flow-min-cost algorithm to solve the problem heuristically and iteratively. The heuristic function is
    customized based on cost_locals, trans_costs, offload_total_costs, local_sat and offload_sat. To approximate the
    optimum with the best effort, optional operation on the raw results is adopted.
    :param server_num: server_num, the server node index starts from 0
    :param user_num: user_num, the user node index starts from server_num
    :param adj_matrix: the adjacent matrix of the di-graph
    :param points: the point feature that indicates the node type, 1 for server and 0 for user
    :param edges: the edge list of the di-graph, each row is a directed edge, the node index starts from 0
    :param local_sat: the satisfaction of theta in local execution, 1 is satisfied and 0 is not
    :param offload_sat: the satisfaction of theta in total offload execution, 1 can be satisfied and 0 cannot
    :param cost_locals: the local execution costs of the edges
    :param trans_locals: the offload transmission costs of the edges
    :param offload_total_costs: the total offload execution costs of the edges
    :param least_ratios: the least ratios of computational resources needed to satisfy the maximum tolerable delays,
                         larger than 0 if can be satisfied and 0 if cannot be satisfied
    :return res_cost: the result cost of the input instance, shape ()
    :return res_edges: the result edge list of the solution, shape (k, 2)
    :return res_adj: the result adjacent matrix of the solution, shape (node_num, node_num)
    :return res_ratios: the result resource allocation ratios of the solution, shape (k,)
    """
    res_edges = np.array([])
    res_adj = np.array([])
    res_ratios = np.array([])
    res_cost = float('inf')
    edge_num = edges.shape[0]

    start_time = time.time()

    # heuristic search for possible weights that indicate the resource allocation ratios
    theta_prior_heuristic = False
    local_offload_ratios = cost_locals / (trans_costs + offload_total_costs)
    heuristic_ratios = np.where(local_offload_ratios <= 1, 0, offload_total_costs / (cost_locals - trans_costs))
    if theta_prior_heuristic is False:
        tmp_sat = local_sat + offload_sat
        heuristic_ratios = np.where(tmp_sat >= 1, least_ratios, heuristic_ratios)
    ws = np.where(heuristic_ratios > 1e-5, heuristic_ratios, 1e-5)
    weights_list = generate_weights_randomly(ws, edges, server_num)
    weights_list = np.where(weights_list > 1e-5, weights_list, 1e-5)  # avoid infinite weights

    for it in range(weights_list.shape[0]):
        weights = trans_costs + offload_total_costs / weights_list[it]

        G = nx.DiGraph()
        # user to user
        edge_dicts = [('1u'+str(i), '2u'+str(i), {'capacity': 1, 'weight': 0}) for i in range(user_num)]
        # user to server
        edge_dicts += [('2u'+str(edges[i][0] - server_num), 's'+str(edges[i][1]), {'capacity': 1, 'weight': int(weights[i] * 1000)}) for i in range(edge_num)]
        # node with source & target
        edge_dicts += [('S', '1u'+str(i), {'capacity': 1, 'weight': 0}) for i in range(user_num)]
        edge_dicts += [('s'+str(i), 'T', {'capacity': user_num, 'weight': 0}) for i in range(server_num)]
        user_locals = []
        for i in range(edge_num):
            if edges[i][0] - server_num >= len(user_locals):
                user_locals.append(cost_locals[i])
        user_locals = np.array(user_locals)
        edge_dicts += [('2u'+str(i), 'T', {'capacity': 1, 'weight': int(user_locals[i] * 1000)}) for i in range(user_num)]
        G.add_edges_from(edge_dicts)

        minCostFlow = nx.max_flow_min_cost(G, 'S', 'T')
        minCost = nx.cost_of_flow(G, minCostFlow)  # raw result cost
        # print(minCostFlow)

        cur_res_adj = np.zeros_like(adj_mat)
        cur_res_edges = []
        for i in range(user_num):
            for end_node, flow in minCostFlow['2u'+str(i)].items():
                if flow == 1:
                    if end_node != 'T':
                        server = int(end_node[1:])
                        cur_res_adj[i + server_num][server] = 1
                        cur_res_edges.append([i + server_num, server])
        cur_res_edges = np.array(cur_res_edges)
        if len(cur_res_edges) == 0:  # no offload at all
            cur_res_cost = np.sum(user_locals)
            cur_edge_weights = np.array([])
        else:
            cur2src_indices = np.array([np.where((edges == row).all(axis=1))[0][0] for row in cur_res_edges])
            cur_edge_weights = weights_list[it][cur2src_indices]
            raw_ratio_sum = np.array([np.sum(cur_edge_weights[np.where(cur_res_edges[:, 1] == i)]) for i in range(server_num)])
            for i in range(server_num):
                server_edges = np.where(cur_res_edges[:, 1] == i)[0]
                if len(server_edges) == 0:
                    continue
                cur_edge_weights[server_edges] += (1 - raw_ratio_sum[i]) * cur_edge_weights[server_edges] / raw_ratio_sum[i]  # !!!!!!!!!!
            cur_res_cost = np.sum(trans_costs[cur2src_indices] + offload_total_costs[cur2src_indices] / cur_edge_weights)
            local_exec = np.array([i for i in range(user_num) if np.all(cur_res_edges[:, 0] != i + server_num)])
            if len(local_exec) > 0:
                cur_res_cost += np.sum(user_locals[local_exec])

        if cur_res_cost < res_cost:
            res_cost = cur_res_cost
            res_edges, res_adj, res_ratios = cur_res_edges, cur_res_adj, cur_edge_weights

    end_time = time.time() - start_time
    # print(f"{weights_list.shape[0]} solved, time cost: {end_time:.4f}s")
    return res_cost, res_edges, res_adj, res_ratios


def generate_explorative_data(directory, sample_num, server_num=None, user_num=None):

    F_t = 33.6e9
    theta = 2.0
    P_t = 0.3
    P_I = 0.15
    kappa = 1e-28
    B = 8e7
    N0 = 7.96159e-13  # xi**2, for SINR calculation

    is_mu, is_sigma, is_low, is_up = 0.65e7, 0.3e7, 0.2e7, 1.1e7
    fl_mu, fl_sigma, fl_low, fl_up = 6.4e9, 5e9, 1e9, 10e9

    res_cost_list, res_edges_list, res_adj_list, res_ratios_list = [], [], [], []
    node_raw_list, edge_raw_list, src_edge_list = [], [], []
    edge_attribute_list = []
    for i in tqdm(range(sample_num)):
        G = generate_graph(server_num, user_num)

        adj_mat = nx.adjacency_matrix(G).toarray()
        points = np.concatenate((np.ones(server_num, dtype=np.int8), np.zeros(user_num, dtype=np.int8)))  # final node feature
        edges = np.array([[list(G.nodes).index(u), list(G.nodes).index(v)] for u, v in G.edges()])
        edge_num = edges.shape[0]

        input_sizes = range_random(is_mu, is_sigma, user_num, is_low, is_up)
        required_cycles = input_sizes * 3e3
        f_locals = range_random(fl_mu, fl_sigma, user_num, fl_low, fl_up)
        alphas = np.random.rand(user_num)
        betas = 1 - alphas

        tau_locals = required_cycles / f_locals
        epsilon_locals = kappa * (f_locals ** 2) * input_sizes
        cost_locals = alphas * tau_locals + betas * epsilon_locals
        local_sat = np.where(theta > tau_locals, 1, 0)

        hs = np.random.rand(edge_num)
        hs_squares = np.array([np.sum(hs[np.where(edges[:, 1] == edges[i, 1])[0]] ** 2) for i in range(edge_num)])
        sinr = P_t * (hs ** 2) / (N0 + P_t * hs_squares)
        r_us = B * np.log2(1 + sinr)

        node_raw_list.append(np.vstack((input_sizes, required_cycles, f_locals, alphas)).T.ravel())
        edge_raw_list.append(hs)
        src_edge_list.append(edges)

        tmp_indices = edges[:, 0] - server_num
        cost_locals = cost_locals[tmp_indices]  # final edge feature
        local_sat = local_sat[tmp_indices]  # final edge feature
        input_sizes = input_sizes[tmp_indices]
        required_cycles = required_cycles[tmp_indices]
        alphas = alphas[tmp_indices]
        betas = 1.0 - alphas
        trans_costs = alphas * input_sizes / r_us + betas * P_t * input_sizes / r_us  # final edge feature
        offload_total_costs = alphas * required_cycles / F_t + betas * P_I * required_cycles / F_t  # final edge feature

        trans_sat = np.where(theta > input_sizes / r_us, 1, 0)
        offload_sat = np.where(theta > input_sizes / r_us + required_cycles / F_t, 1, 0)
        least_ratios = required_cycles / (theta - input_sizes / r_us) / F_t
        least_ratios *= trans_sat
        least_ratios *= offload_sat  # final edge feature

        exist_sat = np.where(local_sat + least_ratios > 0, 1, 0)

        res_cost, res_edges, res_adj, res_ratios = solve_once(server_num, user_num, adj_mat, points, edges, local_sat, offload_sat,
                                                              cost_locals, trans_costs, offload_total_costs, least_ratios)
        res_cost_list.append(res_cost)
        res_edges_list.append(res_edges)
        res_adj_list.append(res_adj)
        res_ratios_list.append(res_ratios)
        edge_attribute_list.append(np.vstack((cost_locals, trans_costs, offload_total_costs, least_ratios, local_sat)).T.ravel())  # same dim with edge_num

    tag = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    res_cost_array = np.array(res_cost_list)
    res_adj_array = np.array(res_adj_list)
    with open(os.path.join(directory, f'{server_num}s{user_num}u_{sample_num}samples_{tag}.txt'), 'w') as f:
        for i in range(sample_num):
            f.write("node " + " ".join(map(str, points)) + " ")
            f.write("edge " + " ".join(map(str, src_edge_list[i].reshape(-1))) + " ")
            f.write("node_raw " + " ".join(map(str, node_raw_list[i])) + " ")
            f.write("edge_raw " + " ".join(map(str, edge_raw_list[i])) + " ")
            f.write("edge_attr " + " ".join(map(str, edge_attribute_list[i])) + " ")
            f.write("gt_edges " + " ".join(map(str, res_edges_list[i].reshape(-1))) + " ")
            f.write("gt_ws " + " ".join(map(str, res_ratios_list[i])) + " ")
            f.write("gt_cost " + str(res_cost_array[i]) + "\n")

    cmf = np.array([F_t, kappa, P_t, P_I, theta, B, N0, is_mu, is_sigma, is_low, is_up, fl_mu, fl_sigma, fl_low, fl_up])
    yml_dict = {}
    yml_keys = ['F_t', 'kappa', 'P_t', 'P_I', 'theta', 'B', 'N0', 's_mu', 's_sigma', 's_low', 's_up', 'fl_mu',
                'fl_sigma', 'fl_low', 'fl_up']
    for i in range(len(yml_keys)):
        yml_dict[yml_keys[i]] = float(cmf[i])
    yml_path = os.path.join(directory, f'{server_num}s{user_num}u_{sample_num}samples_{tag}.yaml')
    with open(yml_path, 'w') as file:
        yaml.dump(yml_dict, file)


def range_random(mu, sigma, size, lower=None, upper=None):
    """
    Generate a random array of normal distribution.
    :param mu: Mean.
    :param sigma: Variance.
    :param size: Array size.
    :param lower: [optional] Lower bound.
    :param upper: [optional] Upper bound.
    :return Raw array.
    """
    # Generate the initial random array
    arr = np.random.normal(mu, sigma, size)
    if lower is None or upper is None:
        return arr

    # Regenerate elements outside the desired range
    while np.any(arr < lower) or np.any(arr > upper):
        arr[arr < lower] = np.random.normal(mu, sigma, np.sum(arr < lower))
        arr[arr > upper] = np.random.normal(mu, sigma, np.sum(arr > upper))
    return arr


if __name__ == "__main__":
    # server_num = random.choice([3, 4, 5])
    # user_num = random.randint(server_num + 1, 4 * server_num - 2)

    generate_explorative_data('msmu-co/', 2000, 20, 61)
