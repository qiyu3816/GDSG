""" MSCO (Multi-Server multi-user Computation Offloading) Graph Dataset"""

import os
import numpy as np
import torch
from torch_geometric.data import Data as GraphData

class MSCOGraphDataset(torch.utils.data.Dataset):

    def __init__(self, data_file, sparse=False):
        self.data_file = data_file
        self.sparse = sparse
        self.file_lines = open(data_file).read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

    def __len__(self):
        return len(self.file_lines)

    def get_example(self, idx):
        # Select sample
        line = self.file_lines[idx]
        # Clear leading/trailing characters
        line = line.strip()

        raw_lst = line.split(' ')

        # Extract nodes
        node_idx = raw_lst.index('node')
        nodes = []
        for i in range(node_idx + 1, len(raw_lst)):
            if raw_lst[i][0].isdigit():
                nodes.append(int(raw_lst[i]))
            else:
                break
        nodes = np.array(nodes)

        # Extract edges
        edge_idx = raw_lst.index('edge')
        edges = []
        for i in range(edge_idx + 1, len(raw_lst), 2):
            if raw_lst[i][0].isdigit():
                edges.append([int(raw_lst[i]), int(raw_lst[i + 1])])
            else:
                break

        # Edge attributes
        edge_attr_idx = raw_lst.index('edge_attr')
        for i in range(edge_attr_idx + 1, len(raw_lst), 5):
            if raw_lst[i][0].isdigit():
                edges[(i - edge_attr_idx) // 5] += [float(raw_lst[i]), float(raw_lst[i + 1]), float(raw_lst[i + 2]), float(raw_lst[i + 3]), float(raw_lst[i + 4])]
            else:
                break
        edges = np.array(edges)

        # Ground truth edges
        gt_edge_idx = raw_lst.index('gt_edges')
        gt_edges = []
        for i in range(gt_edge_idx + 1, len(raw_lst), 2):
            if raw_lst[i][0].isdigit():
                gt_edges.append([int(raw_lst[i]), int(raw_lst[i + 1])])
            else:
                break
        gt_edges = np.array(gt_edges)

        # Ground truth edge weights
        gt_edge_weights_idx = raw_lst.index('gt_ws')
        gt_edge_ws = []
        for i in range(gt_edge_weights_idx + 1, len(raw_lst)):
            if raw_lst[i][0].isdigit():
                gt_edge_ws.append(float(raw_lst[i]))
            else:
                break
        gt_edge_ws = np.array(gt_edge_ws)

        # Ground truth cost
        gt_cost_idx = raw_lst.index('gt_cost')
        gt_cost = float(raw_lst[gt_cost_idx + 1])

        return nodes, edges, gt_edges, gt_edge_ws, gt_cost

    def add_edges(self, edges, server_num, default_feature=[0, 10e3, 10e3, 10e3, 2, 1]):
        """
        Since the batch stack in training requires isomorphic graphs, padding is performed on the edges that actually
        exist in each sample to make the user and the server fully connected. However, the features of these padding
        edges will be set to default values. The first dimension of the default value indicates whether the edge
        actually exists, 0 if it does not exist, and 1 if it does.
        """
        edges = np.insert(edges, 2, 1, axis=1)
        start_nodes = np.unique(edges[:, 0])
        comp_target = server_num
        for node in start_nodes:
            node_s_edges = edges[np.abs(edges[:, 0] - node) < 1e-3]

            if node_s_edges.shape[0] < comp_target:
                existing_end_nodes = node_s_edges[:, 1]
                possible_end_nodes = np.setdiff1d(np.arange(server_num), existing_end_nodes)
                for end_node in possible_end_nodes:
                    new_edge = np.array([node, end_node] + default_feature)
                    edges = np.vstack((edges, new_edge))
        first_column = edges[:, 0]
        sorted_indices = np.argsort(first_column)
        edges = edges[sorted_indices]
        return edges

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:
            Sparse
            - idx: index of the sample
            - graph_data_input: x(V), edge_index(U*comp_target), comp_target=4 if server_num > 3 else 3
                                edge_attr(U*comp_target, 6), 6 = [edge_real_exit 1 dim + 5 data_dim]
            - local_costs: (U)
            - gt_edge_index: (U*comp_target)
            - gt_edge_ws: (U*comp_target)
            - gt_cost: float item tensor
            Dense
            - idx: index of the sample
            - nodes: (V)
            - adj_matrix: (6, V, V), 6 = [edge_real_exit 1 dim + 5 data_dim]
            - local_costs: (U)
            - gt_adj_matrix: (V, V)
            - gt_adj_ws: (V, V)
            - gt_cost: float item tensor

        """
        nodes, edges, gt_edges, gt_edge_ws, gt_cost = self.get_example(idx)
        server_num = int(np.sum(nodes))
        user_num = len(nodes) - server_num

        gt_adj_matrix = np.zeros((len(nodes), len(nodes)))
        for edge in gt_edges:
            gt_adj_matrix[int(edge[0]), int(edge[1])] = 1
        gt_adj_ws = np.zeros((len(nodes), len(nodes)))
        for i in range(gt_edges.shape[0]):
            gt_adj_ws[int(gt_edges[i][0]), int(gt_edges[i][1])] = gt_edge_ws[i]

        if self.sparse:
            # Return a sparse graph
            _, indices = np.unique(edges[:, 0], return_index=True)
            local_costs = edges[indices, 2]
            edges = self.add_edges(edges, server_num)
            graph_data_input = GraphData(x=torch.from_numpy(nodes).long(),
                                   edge_index=torch.from_numpy(edges[:, :2].T).long(),
                                   edge_attr=torch.from_numpy(edges[:, 2:]).float())
            gt_edge_index = np.zeros_like(edges[:, 0], dtype=int)
            gt_edge_ws = np.zeros_like(edges[:, 0], dtype=float)
            for i in range(edges.shape[0]):
                start_node, end_node = edges[i, :2].astype(int)
                if gt_adj_matrix[start_node, end_node] == 1:
                    gt_edge_index[i] = 1
                    gt_edge_ws[i] = gt_adj_ws[start_node, end_node]
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                graph_data_input,
                torch.from_numpy(local_costs).float(),
                torch.from_numpy(gt_edge_index).float(),
                torch.from_numpy(gt_edge_ws).float(),
                torch.from_numpy(np.array([gt_cost], dtype=np.float32))
            )
        else:
            # Return a densely connected graph
            adj_matrix = np.zeros((6, len(nodes), len(nodes)))
            for edge in edges:
                adj_matrix[0, int(edge[0]), int(edge[1])] = 1
                adj_matrix[1:, int(edge[0]), int(edge[1])] = edge[2:]

            first_one_indices = np.argmax(adj_matrix[0, -user_num:, :server_num], axis=-1)
            local_costs = np.zeros_like(adj_matrix[1, -user_num:, 0])
            for i in range(local_costs.shape[0]):
                local_costs[i] = adj_matrix[1, server_num + i, first_one_indices[i]]

            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                torch.from_numpy(nodes).long(),
                torch.from_numpy(adj_matrix).float(),
                torch.from_numpy(local_costs).float(),
                torch.from_numpy(gt_adj_matrix).float(),
                torch.from_numpy(gt_adj_ws).float(),
                torch.from_numpy(np.array([gt_cost], dtype=np.float32))
            )

    def wrap_label2str(self, label):
        gt_edges, gt_ws, gt_cost = label
        gt_edges_str = 'gt_edges '
        gt_ws_str = 'gt_ws '
        user_num, server_num = gt_edges.shape[0], gt_edges.shape[1]
        for i in range(user_num):
            for j in range(server_num):
                if gt_edges[i, j].int() == 1:
                    gt_edges_str += f'{i + server_num} {j} '
                    gt_ws_str += f'{gt_ws[i, j]} '
        gt_cost_str = f'gt_cost {gt_cost}'
        label_str = f'{gt_edges_str}{gt_ws_str}{gt_cost_str}'
        return label_str

    def replace_label(self, idx, new_label):
        line = self.file_lines[idx]
        label_str = self.wrap_label2str(new_label)
        index = line.find('gt_edges')
        line = line[:index] + label_str
        self.file_lines[idx] = line

    def re_dump(self, idx=-1):
        if idx > 0:
            out_lines = self.file_lines[:idx]
        else:
            idx = len(self.file_lines)
            out_lines = self.file_lines
        dir_part = os.path.dirname(self.data_file)
        filename = os.path.basename(self.data_file)
        new_filename = f"refine{idx}_" + filename
        new_filepath = os.path.join(dir_part, new_filename)
        with open(new_filepath, 'w') as f:
            for line in out_lines:
                f.write(line + '\n')
        print(f"Re-dump done! Refined dataset -> {new_filepath}")
