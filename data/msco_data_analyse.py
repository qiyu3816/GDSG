import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from difusco.co_datasets.msco_graph_dataset import MSCOGraphDataset


dataset = MSCOGraphDataset('D:/Codes/DT4SG/data/msmu-co/4server/4s10u_10000samples_20240604221408.txt')
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
server_num = 4

local_cost_max, local_cost_min = 0, float('inf')
trans_cost_max, trans_cost_min = 0, float('inf')
offlo_cost_max, offlo_cost_min = 0, float('inf')
gt_cost_max, gt_cost_min = 0, float('inf')
cnt_extreme_trans = 0
for batch in tqdm(data_loader):
    _, nodes, adj_matrix, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
    for i in range(adj_matrix.shape[0]):
        for j in range(server_num, adj_matrix.shape[2]):
            for k in range(server_num):
                if adj_matrix[i, 0, j, k] == 1:
                    local_cost_max = max(adj_matrix[i, 1, j, k], local_cost_max)
                    local_cost_min = min(adj_matrix[i, 1, j, k], local_cost_min)
                    trans_cost_max = max(adj_matrix[i, 2, j, k], trans_cost_max)
                    if adj_matrix[i, 2, j, k] > 32:
                        # print(adj_matrix[i, 1, j, k], adj_matrix[i, 3, j, k], "terrible instance")
                        cnt_extreme_trans += 1
                    trans_cost_min = min(adj_matrix[i, 2, j, k], trans_cost_min)
                    offlo_cost_max = max(adj_matrix[i, 3, j, k], offlo_cost_max)
                    offlo_cost_min = min(adj_matrix[i, 3, j, k], offlo_cost_min)
    gt_cost_max = max(torch.max(gt_cost), gt_cost_max)
    gt_cost_min = min(torch.min(gt_cost), gt_cost_min)

print(cnt_extreme_trans)  # 10000->178  20000->409  80000->1622
print("{:.4f}, {:.4f}".format(local_cost_min.item(), local_cost_max.item()))
print("{:.4f}, {:.4f}".format(trans_cost_min.item(), trans_cost_max.item()))
print("{:.4f}, {:.4f}".format(offlo_cost_min.item(), offlo_cost_max.item()))
print("{:.4f}, {:.4f}".format(gt_cost_min.item(), gt_cost_max.item()))
