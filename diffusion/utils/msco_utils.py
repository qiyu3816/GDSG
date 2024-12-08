import torch

local_cost_offload_threshold = 25

def calc_cost_upper(server_num, user_num):
    cost_upper = -4.98473058e+00 * server_num + 3.07564767e+00 * user_num + 1.39866115e-02 * server_num * server_num + 1.73116225e-03 * user_num * user_num + 2.39982902e+01
    return cost_upper if cost_upper > 0 else 0.1


def cost_calc_once(adj_matrix, pred_adj_mat, pred_adj_ws):
    """
    Calculate the cost based on the **completely valid** predicted adjacency matrix and the predicted adjacency weights for a batch.
    Args:
        adj_matrix: (B, 6, U, S)
        pred_adj_mat: (B, U, S)
        pred_adj_ws: (B, U, S)

    Returns:

    """
    offload_decision = torch.sum(pred_adj_mat, dim=-1)

    pred_adj_ws = torch.where(pred_adj_ws > 1e-5, pred_adj_ws, 1e-5)
    offload_cost = pred_adj_mat * adj_matrix[:, 2] + pred_adj_mat * adj_matrix[:, 3] / pred_adj_ws
    pred_adj_ws = pred_adj_ws * pred_adj_mat   # ensure valid
    offload_cost = torch.sum(offload_cost, dim=(1, 2))

    first_one_indices = torch.argmax(adj_matrix[:, 0].squeeze(1), dim=-1)
    local_costs = torch.zeros_like(adj_matrix[:, 1, :, 0])
    for i in range(local_costs.shape[0]):
        for j in range(local_costs.shape[1]):
            local_costs[i, j] = adj_matrix[i, 1, j, first_one_indices[i][j]]
    local_cost = torch.sum((1 - offload_decision) * local_costs, dim=-1)
    pred_cost = local_cost + offload_cost

    return pred_cost


def validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws):
    """
    Deal with the validity of the **raw** predicted adjacency matrix and adjacency weights.
    Args:
        adj_matrix: (B, 6, U, S)
        pred_adj_mat: (B, U, S)
        pred_adj_ws: (B, U, S)

    Returns:

    """
    # whether to offload and check validity
    offload_decision = torch.sum(pred_adj_mat, dim=-1)
    invalid_user_indices = torch.argwhere(offload_decision > 1)
    tmp_mat = pred_adj_mat * pred_adj_ws
    for i in range(invalid_user_indices.shape[0]):  # deal multiple offload
        keep_offload = torch.argmax(tmp_mat[invalid_user_indices[i][0], invalid_user_indices[i][1], :])
        pred_adj_mat[invalid_user_indices[i][0], invalid_user_indices[i][1], :] = 0
        pred_adj_mat[invalid_user_indices[i][0], invalid_user_indices[i][1], keep_offload] = 1
    invalid_user_indices = torch.argwhere(pred_adj_mat - adj_matrix[:, 0] > 0)
    for i in range(invalid_user_indices.shape[0]):  # deal no connection offload
        valid_optional_indices = torch.argwhere(torch.abs(adj_matrix[invalid_user_indices[i][0], 0, invalid_user_indices[i][1]] - 1) < 1e-3)
        idx = valid_optional_indices[torch.randint(0, valid_optional_indices.shape[0], (1,))][0].item()
        pred_adj_mat[invalid_user_indices[i][0], invalid_user_indices[i][1], invalid_user_indices[i][2]] = 0
        pred_adj_mat[invalid_user_indices[i][0], invalid_user_indices[i][1], idx] = 1
        pred_adj_ws[invalid_user_indices[i][0], invalid_user_indices[i][1], idx] = pred_adj_ws[invalid_user_indices[i][0], invalid_user_indices[i][1], invalid_user_indices[i][2]]

    # avg allocation or weighted allocation
    pred_adj_ws = ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)

    return pred_adj_mat, pred_adj_ws


def ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws):
    # pred_adj_ws = pred_adj_ws * pred_adj_mat
    # offload_sum = torch.sum(pred_adj_ws, dim=1).unsqueeze(1)
    # offload_sum = torch.where(offload_sum > 1e-5, offload_sum, 1e-5)
    # pred_adj_ws = pred_adj_ws / offload_sum
    offload_cost_sum = torch.sum(adj_matrix[:, 3] * pred_adj_mat, dim=1)
    offload_cost_sum = torch.where(offload_cost_sum > 1e-5, offload_cost_sum, 1e-5)
    ws_ratio = adj_matrix[:, 3] * pred_adj_mat / offload_cost_sum[:, None, :]
    pred_adj_ws = ws_ratio
    return pred_adj_ws


def optional_iteration(adj_matrix, pred_adj_mat, pred_adj_ws, iterations=50):
    """
    Iteratively randomly adjust the predicted adjacency matrix and the predicted adjacency weights for a batch
    to get better solutions.
    Args:
        adj_matrix: (B, 6, U, S)
        pred_adj_mat: (B, U, S)
        pred_adj_ws: (B, U, S)
        iterations:

    Returns:

    """
    # 1. The two 0-1 list building for those must be executed locally and must be offloaded
    must_local = torch.where(adj_matrix[:, 1] > adj_matrix[:, 2] + adj_matrix[:, 3], 1, 0) * adj_matrix[:, 0]
    must_local = torch.where(torch.sum(must_local, dim=-1) == 0, 1, 0)

    must_offload = torch.sum(torch.where(adj_matrix[:, 1] > local_cost_offload_threshold, 1, 0), dim=-1)
    must_offload = torch.where(must_offload > 0, 1, 0)
    no_must = 1 - (must_local + must_offload)

    for i in range(must_local.shape[0]):
        for j in range(must_local.shape[1]):
            if must_local[i, j] == 1:
                pred_adj_mat[i, j] = 0
    for i in range(must_offload.shape[0]):
        for j in range(must_offload.shape[1]):
            if must_offload[i, j] == 1 and torch.all(pred_adj_mat[i, j] == 0):
                valid_optional_indices = torch.argwhere(torch.abs(adj_matrix[i, 0, j] - 1) < 1e-3)
                idx = valid_optional_indices[torch.randint(0, valid_optional_indices.shape[0], (1,))][0].item()
                pred_adj_mat[i, j, idx] = 1
    pred_cost = cost_calc_once(adj_matrix, pred_adj_mat, pred_adj_ws)

    # 2. Several iterations of randomization
    # First, randomly select the user from current offloadings or the no-must, and then select the conversion mode (offload server switch/offload or local switch).
    # The final proportional allocation is heuristic (modular)
    for ite in range(iterations):
        if ite % 2 == 0:  # randomly change the offload server for one user
            offload_decision = torch.sum(pred_adj_mat, dim=-1)
            for i in range(offload_decision.shape[0]):
                offload_decision[i] = torch.where(torch.sum(offload_decision[i]) > 0, offload_decision[i], 1)
            prob_dist = offload_decision.float()
            selected_user_indices = torch.multinomial(prob_dist, 1)
            for i in range(pred_adj_mat.shape[0]):
                j = selected_user_indices[i].item()
                if torch.any(pred_adj_mat[i, j] == 1):
                    before_server = torch.nonzero(pred_adj_mat[i, j].eq(1), as_tuple=True)[0][0]
    
                    valid_optional_indices = torch.argwhere(torch.abs(adj_matrix[i, 0, j] - 1) < 1e-3)
                    idx = valid_optional_indices[torch.randint(0, valid_optional_indices.shape[0], (1,))][0]
                    pred_adj_mat[i, j] = 0
                    pred_adj_mat[i, j, idx] = 1
                    pred_adj_ws = ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)
    
                    new_cost = cost_calc_once(adj_matrix[i].unsqueeze(0), pred_adj_mat[i].unsqueeze(0), pred_adj_ws[i].unsqueeze(0))
                    if new_cost < pred_cost[i]:
                        pred_cost[i] = new_cost
                    else:
                        pred_adj_mat[i, j] = 0
                        pred_adj_mat[i, j, before_server] = 1
                        pred_adj_ws = ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)
        else:  # randomly shift one no-must user's state
            offload_decision = torch.sum(pred_adj_mat, dim=-1)
            prob_dist = no_must.float()
            for i in range(prob_dist.shape[0]):
                prob_dist[i] = torch.where(torch.sum(prob_dist[i]) > 1e3, prob_dist[i], 1)
            selected_user_indices = torch.multinomial(prob_dist, 1)
            for i in range(pred_adj_mat.shape[0]):
                if offload_decision[i, selected_user_indices[i]] > 0:
                    before_server = torch.nonzero(pred_adj_mat[i, selected_user_indices[i]].eq(1), as_tuple=True)[-1][0]
                    pred_adj_mat[i, selected_user_indices[i]] = 0
                    new_cost = cost_calc_once(adj_matrix[i].unsqueeze(0), pred_adj_mat[i].unsqueeze(0), pred_adj_ws[i].unsqueeze(0))
                    if new_cost < pred_cost[i]:
                        pred_cost[i] = new_cost
                    else:
                        pred_adj_mat[i, selected_user_indices[i]] = 0
                        pred_adj_mat[i, selected_user_indices[i], before_server] = 1
                        pred_adj_ws = ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)
                else:
                    valid_optional_indices = torch.argwhere(torch.abs(adj_matrix[i, 0, selected_user_indices[i][0]] - 1) < 1e-3)
                    idx = valid_optional_indices[torch.randint(0, valid_optional_indices.shape[0], (1,))][0].item()
                    pred_adj_mat[i, selected_user_indices[i].item(), idx] = 1
                    pred_adj_ws = ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)
                    new_cost = cost_calc_once(adj_matrix[i].unsqueeze(0), pred_adj_mat[i].unsqueeze(0), pred_adj_ws[i].unsqueeze(0))
                    if new_cost < pred_cost[i]:
                        pred_cost[i] = new_cost
                    else:
                        pred_adj_mat[i, selected_user_indices[i].item()] = 0
                        pred_adj_ws = ws_validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)
    return pred_cost, pred_adj_mat, pred_adj_ws


def cost_calc(nodes, adj_matrix, pred_adj_mat, pred_adj_ws, proprocess=False):
    """
    Calculate the cost based on the predicted adjacency matrix and the predicted adjacency weights for a batch.
    Only for the same server and user num setting.
    Args:
        nodes: (B, V)
        adj_matrix: (B, 6, V, V)
        pred_adj_mat: (B, V, V)
        pred_adj_ws: (B, V, V)

    Returns:

    """
    with torch.no_grad():
        server_num = torch.sum(nodes, dim=1)[0].item()
        user_num = nodes.shape[1] - server_num
        local_cost_offload_threshold = calc_cost_upper(server_num, user_num) * 0.9

        adj_matrix = adj_matrix.clone()[:, :, server_num:, :server_num]
        pred_adj_mat = pred_adj_mat.clone()[:, server_num:, :server_num]
        pred_adj_ws = pred_adj_ws.clone()[:, server_num:, :server_num]

        pred_adj_mat, pred_adj_ws = validity_deal(adj_matrix, pred_adj_mat, pred_adj_ws)

    if proprocess:
        return optional_iteration(adj_matrix, pred_adj_mat, pred_adj_ws)
    return cost_calc_once(adj_matrix, pred_adj_mat, pred_adj_ws), pred_adj_mat, pred_adj_ws
