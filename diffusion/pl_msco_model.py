""" Lightning model for training and evaluating DIFUSCO MSCO models. """

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from co_datasets.msco_graph_dataset import MSCOGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.msco_utils import cost_calc, validity_deal, calc_cost_upper


def count_rows(a, b):
    a_sum = np.sum(a, axis=-1)
    b_sum = np.sum(b, axis=-1)

    condition1 = (a_sum == 0) & (b_sum != 0)
    condition2 = (a_sum > 0) & (b_sum == 0)

    count1 = np.sum(condition1)
    count2 = np.sum(condition2)

    return count1, count2


class MSCOModel(COMetaModel):

    def __init__(self, param_args=None):
        super(MSCOModel, self).__init__(param_args=param_args, node_feature_only=False)

        self.train_dataset = MSCOGraphDataset(data_file=os.path.join(self.args.storage_path, self.args.training_split), sparse=self.sparse)

        self.test_dataset = MSCOGraphDataset(data_file=os.path.join(self.args.storage_path, self.args.test_split), sparse=self.sparse)

        self.validation_dataset = MSCOGraphDataset(data_file=os.path.join(self.args.storage_path, self.args.validation_split), sparse=self.sparse)

        self.cost_upper = None
        self.cost_lower = 0

        self.cls_threshold = 0.9

        self.test_metrics = None
        self.best_solved_costs = []
        self.best_solved_solutions = []

    def forward(self, nodes, t, xt, edges):
        return self.model(nodes, t, xt, edges)

    def normalize_input(self, adj_matrix, server_num, user_num):
        self.cost_upper = calc_cost_upper(server_num, user_num)
        if self.sparse:
            adj_matrix[3:6, :] = ((adj_matrix[3:6, :] - self.cost_lower) / (self.cost_upper - self.cost_lower)).clamp(0, 1)
        else:
            adj_matrix[:, 1:4, :, :] = ((adj_matrix[:, 1:4, :, :] - self.cost_lower) / (self.cost_upper - self.cost_lower)).clamp(0, 1)

    def categorical_training_step(self, batch, batch_idx):
        if self.sparse:
            _, graph_data, local_costs, gt_adj_matrix, _, gt_cost = batch
            nodes = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
            adj_matrix = torch.cat((edge_index, edge_attr.T), dim=0)
            batch_size = gt_cost.shape[0]
            num_edges_per_graph = gt_adj_matrix.shape[1]
            server_num = (torch.sum(nodes) // batch_size).item()
            user_num = (torch.sum(1 - nodes) // batch_size).item()
        else:
            _, nodes, adj_matrix, local_costs, gt_adj_matrix, _, gt_cost = batch
            server_num = torch.sum(nodes[0]).item()
            user_num = torch.sum(1 - nodes[0]).item()
        self.normalize_input(adj_matrix, server_num, user_num)
        t = np.random.randint(1, self.diffusion.T + 1, gt_cost.shape[0]).astype(int)

        gt_adj_onehot = F.one_hot(gt_adj_matrix.long(), num_classes=2).float()
        if self.sparse:
            gt_adj_onehot = gt_adj_onehot.unsqueeze(1)
        xt = self.diffusion.sample(gt_adj_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        if self.sparse:
            t = torch.from_numpy(t).float()
            t = t.repeat_interleave(num_edges_per_graph)
            xt = xt.reshape(-1)
            gt_adj_matrix = gt_adj_matrix.reshape(-1)
        else:
            t = torch.from_numpy(t).float().view(gt_adj_matrix.shape[0])

        x0_pred = self.forward(
            nodes.float().to(gt_adj_matrix.device),
            t.float().to(gt_adj_matrix.device),
            xt.float().to(gt_adj_matrix.device),
            adj_matrix.float().to(gt_adj_matrix.device),
        )

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, gt_adj_matrix.long())
        self.log("train/loss", loss)
        return loss

    def gaussian_training_step(self, batch, batch_idx):
        if self.sparse:
            _, graph_data, local_costs, _, gt_adj_ws, gt_cost = batch
            nodes = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
            adj_matrix = torch.cat((edge_index, edge_attr.T), dim=0)
            batch_size = gt_cost.shape[0]
            num_edges_per_graph = gt_adj_ws.shape[1]
            server_num = (torch.sum(nodes) // batch_size).item()
            user_num = (torch.sum(1 - nodes) // batch_size).item()
        else:
            _, nodes, adj_matrix, local_costs, _, gt_adj_ws, gt_cost = batch
            server_num = torch.sum(nodes[0]).item()
            user_num = torch.sum(1 - nodes[0]).item()
        self.normalize_input(adj_matrix, server_num, user_num)
        t = np.random.randint(1, self.diffusion_steps + 1, gt_cost.shape[0]).astype(int)

        xt, epsilon = self.diffusion.sample(gt_adj_ws, t)

        t = torch.from_numpy(t).float().view(gt_adj_ws.shape[0])
        if self.sparse:
            t = t.repeat_interleave(num_edges_per_graph)
            xt = xt.reshape(-1)
            epsilon = epsilon.reshape(-1)

        epsilon_pred = self.forward(
            nodes.float().to(gt_adj_ws.device),
            t.float().to(gt_adj_ws.device),
            xt.float().to(gt_adj_ws.device),
            adj_matrix.float().to(gt_adj_ws.device),
        )

        loss_func = F.mse_loss
        loss = loss_func(epsilon_pred.squeeze(), epsilon.float())
        self.log("train/loss", loss)
        return loss

    def hybrid_training_step(self, batch, batch_idx):
        if self.sparse:
            _, graph_data, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
            nodes = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
            adj_matrix = torch.cat((edge_index, edge_attr.T), dim=0)
            batch_size = gt_cost.shape[0]
            num_edges_per_graph = gt_adj_matrix.shape[1]
            server_num = (torch.sum(nodes) // batch_size).item()
            user_num = (torch.sum(1 - nodes) // batch_size).item()
        else:
            _, nodes, adj_matrix, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
            server_num = torch.sum(nodes[0]).item()
            user_num = torch.sum(1 - nodes[0]).item()
        self.normalize_input(adj_matrix, server_num, user_num)
        t = np.random.randint(1, self.diffusion_steps + 1, gt_cost.shape[0]).astype(int)

        gt_adj_onehot = F.one_hot(gt_adj_matrix.long(), num_classes=2).float()
        if self.sparse:
            gt_adj_onehot = gt_adj_onehot.unsqueeze(1)
        xt_reg, epsilon = self.gau_diffusion.sample(gt_adj_ws, t)
        xt_cls = self.cat_diffusion.sample(gt_adj_onehot, t)
        xt_cls = xt_cls * 2 - 1
        xt_cls = xt_cls * (1.0 + 0.05 * torch.rand_like(xt_cls))

        t = torch.from_numpy(t).float().view(gt_adj_matrix.shape[0])
        if self.sparse:
            t = t.repeat_interleave(num_edges_per_graph)
            xt_cls = xt_cls.reshape(-1)
            gt_adj_matrix = gt_adj_matrix.reshape(-1)
            xt_reg = xt_reg.reshape(-1)
            epsilon = epsilon.reshape(-1)

        xt = torch.cat((xt_cls.unsqueeze(1), xt_reg.unsqueeze(1)), dim=1)
        x0_pred = self.forward(
            nodes.float().to(gt_adj_matrix.device),
            t.float().to(gt_adj_matrix.device),
            xt.float().to(gt_adj_matrix.device),
            adj_matrix.float().to(gt_adj_matrix.device),
        )

        if self.sparse:
            x0_pred_cls = x0_pred[:, :2]
            x0_pred_reg = x0_pred[:, -1]
        else:
            x0_pred_cls = x0_pred[:, :2, :, :]
            x0_pred_reg = x0_pred[:, -1, :, :]

        loss_cls = nn.CrossEntropyLoss()
        loss_reg = F.mse_loss

        if self.args.grad_calculate:
            loss_cls_val = loss_cls(x0_pred_cls, gt_adj_matrix.long())
            loss_reg_val = loss_reg(x0_pred_reg, epsilon.float())
            # gradients for classification loss
            self.zero_grad()
            loss_cls_val.backward(retain_graph=True)
            grad_cls = {name: param.grad.clone() for name, param in self.named_parameters() if param.grad is not None}
            self.zero_grad()

            # gradients for regression loss
            self.zero_grad()
            loss_reg_val.backward(retain_graph=True)
            grad_reg = {name: param.grad.clone() for name, param in self.named_parameters() if param.grad is not None}
            self.zero_grad()

            # Save gradients to file
            with open(f'{server_num}s{user_num}u_grads.txt', 'a') as f:
                f.write(f'Epoch: {self.current_epoch}\n')
                for name in grad_reg:
                    f.write(
                        f'Parameter: {name}, \nClassification loss gradient: \n{grad_cls[name]}, \nRegression loss gradient: \n{grad_reg[name]}\n\n')

            loss = loss_cls(x0_pred_cls, gt_adj_matrix.long())
            self.log("train/cls_loss", loss)
            if self.frozen or self.args.freeze_epoch <= 0:
                reg_loss = loss_reg(x0_pred_reg, epsilon.float())
                loss += reg_loss
                self.log("train/reg_loss", reg_loss)
        else:
            loss = loss_cls(x0_pred_cls, gt_adj_matrix.long())
            self.log("train/cls_loss", loss)
            if self.frozen or self.args.freeze_epoch <= 0:
                reg_loss = loss_reg(x0_pred_reg, epsilon.float())
                loss += reg_loss
                self.log("train/reg_loss", reg_loss)
        return loss

    def on_train_epoch_end(self):
        if self.diffusion_type == 'hybrid':
            if self.args.freeze_epoch <= 0:
                pass
            if self.current_epoch == self.args.freeze_epoch:
                self.frozen = True
                self.model.custom_freeze()
                self.print("The hybrid diffusion model backbone frozen. Regressive loss launched.")

    def training_step(self, batch, batch_idx):
        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            return self.categorical_training_step(batch, batch_idx)
        elif self.diffusion_type == 'hybrid':
            return self.hybrid_training_step(batch, batch_idx)

    def categorical_denoise_step(self, xt_cls, nodes, t, target_t, adj_matrix, device):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                nodes.float().to(device),
                t.float().to(device),
                xt_cls.float().to(device),
                adj_matrix.float().to(device),
            )

        if self.sparse:
            x0_pred_prob = x0_pred[:, :2].reshape((1, self.args.parallel_sampling * self.user_num, -1, 2)).softmax(dim=-1)
        else:
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        xt_cls = self.categorical_posterior(target_t, t, x0_pred_prob, xt_cls)
        return xt_cls

    def gaussian_denoise_step(self, xt_reg, nodes, t, target_t, adj_matrix, device):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                nodes.float().to(device),
                t.float().to(device),
                xt_reg.float().to(device),
                adj_matrix.float().to(device),
            )

            xt_reg = self.gaussian_posterior(target_t, t, pred, xt_reg)
        return xt_reg

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        The current test only supports the msco problem, that is, the hybrid diffusion mode.
        For other problems, you need write your own test_step.
        If parallel_sampling > 1, batch_size needs to be 1.
        If batch_size > 1 and you want to repeat sampling, you can only set sequential_sampling > 1.
        !In fact, given the purpose of the code functionality, we recommend using only test_batch=1 in your tests to avoid weird issues.!
        """
        if self.sparse:
            _, graph_data, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
            nodes = graph_data.x
            edge_index = graph_data.edge_index
            edge_attr = graph_data.edge_attr
            adj_matrix = torch.cat((edge_index, edge_attr.T), dim=0)
            batch_size = gt_cost.shape[0]
            num_edges_per_graph = gt_adj_matrix.shape[1]
            server_num = torch.sum(nodes) // batch_size
            user_num = torch.sum(1 - nodes) // batch_size
            self.server_num, self.user_num = server_num, user_num
        else:
            _, nodes, adj_matrix, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
            server_num = torch.sum(nodes, dim=1)[0].item()
            user_num = torch.sum(1 - nodes, dim=1)[0].item()
            batch_size = nodes.shape[0]
        adj_matrix_bak = adj_matrix.clone()
        self.normalize_input(adj_matrix, server_num, user_num)
        device = adj_matrix.device

        if self.args.parallel_sampling > 1:
            assert batch_size == 1  # test_batch == 1 required for parallel_sampling > 1
            if self.sparse:
                nodes = nodes.repeat(self.args.parallel_sampling)
                parallel_edge_index = self.duplicate_edge_index(adj_matrix[:2], server_num + user_num, device)
                adj_matrix = adj_matrix.repeat(1, self.args.parallel_sampling)
                adj_matrix[:2] = parallel_edge_index
                adj_matrix_bak = adj_matrix_bak.repeat(1, self.args.parallel_sampling)
            else:
                nodes = nodes.repeat(self.args.parallel_sampling, 1)
                adj_matrix = adj_matrix.repeat(self.args.parallel_sampling, 1, 1, 1)
                adj_matrix_bak = adj_matrix_bak.repeat(self.args.parallel_sampling, 1, 1, 1)

        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                          T=self.diffusion_steps, inference_T=steps)

        ############################
        ###### raw denoising #######
        solved_cost = []
        solved_solutions = []
        if self.test_metrics is None:
            self.test_metrics = {"test/selection_acc": [], "test/exceed_ratio": []}
        for _ in range(self.args.sequential_sampling):
            if self.diffusion_type == 'hybrid':
                if self.sparse:
                    xt = torch.randn_like(torch.cat((gt_adj_matrix, gt_adj_matrix), dim=0).repeat(1, self.args.parallel_sampling)).float()
                    xt = xt.T
                else:
                    xt = torch.randn_like(torch.cat((gt_adj_matrix.unsqueeze(1), gt_adj_ws.unsqueeze(1)), dim=1).repeat(self.args.parallel_sampling, 1, 1, 1)).float()
                xt[:, 0] = (xt[:, 0] > 0).long()
            elif self.diffusion_type == 'gaussian':
                if self.sparse:
                    xt = torch.randn_like(gt_adj_ws.repeat(1, self.args.parallel_sampling).float())
                else:
                    xt = torch.randn_like(gt_adj_ws.repeat(self.args.parallel_sampling, 1, 1).float())
                    xt = xt.unsqueeze(1)
            else:
                if self.sparse:
                    xt = torch.randn_like(gt_adj_matrix.repeat(1, self.args.parallel_sampling)).float()
                else:
                    xt = torch.randn_like(gt_adj_matrix.repeat(self.args.parallel_sampling, 1, 1).float())
                xt = (xt > 0).long()

            # Diffusion iterations
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)

                if self.diffusion_type == 'gaussian':
                    if xt.shape[0] != adj_matrix.shape[1]:
                        xt = xt.T
                    xt = self.gaussian_denoise_step(xt, nodes, t1, t2, adj_matrix, device)
                elif self.diffusion_type == 'categorical':
                    if xt.shape[0] != adj_matrix.shape[1]:
                        xt = xt.T
                    xt = self.categorical_denoise_step(xt, nodes, t1, t2, adj_matrix, device)
                else:  # hybrid
                    t1 = torch.from_numpy(t1).view(1)
                    if self.sparse:
                        x0_pred = self.forward(
                            nodes.float().to(device),
                            t1.float().to(device),
                            xt.float().to(device),
                            adj_matrix.float().to(device))
                        x0_pred_prob = x0_pred[:, :2].reshape(
                            (1, self.args.parallel_sampling * user_num, -1, 2)).softmax(dim=-1)
                        x0_pred_reg = x0_pred[:, -1]
                        xt_cls = self.categorical_posterior(t2, t1, x0_pred_prob, xt[:, 0])
                        xt_reg = self.gaussian_posterior(t2, t1, x0_pred_reg, xt[:, 1])
                    else:
                        x0_pred = self.forward(
                            nodes.float().to(device),
                            t1.float().to(device),
                            xt.float().to(device),
                            adj_matrix.float().to(device))
                        x0_pred_prob = x0_pred.permute((0, 2, 3, 1))[:, :, :, :2].contiguous().softmax(dim=-1)
                        xt_cls = self.categorical_posterior(t2, t1, x0_pred_prob, xt[:, 0, :, :])
                        x0_pred_reg = x0_pred[:, -1, :, :]
                        xt_reg = self.gaussian_posterior(t2, t1, x0_pred_reg, xt[:, 1, :, :])
                    xt = torch.cat((xt_cls.unsqueeze(1), xt_reg.unsqueeze(1)), dim=1)

            if self.sparse:
                if self.diffusion_type == 'hybrid' or self.diffusion_type == 'categorical':
                    pred_edge_index = np.where(xt[:, 0].float().cpu().detach().numpy() > self.cls_threshold, 1, 0)
                    correct_cnt = pred_edge_index * gt_adj_matrix.repeat(1, self.args.parallel_sampling).long().cpu().detach().numpy()
                    acc = np.mean(correct_cnt)
                edge_index_bak = adj_matrix_bak.clone()
                xt = self.pred_edge2adj(xt, adj_matrix_bak[:2], self.args.parallel_sampling, server_num, user_num, device)
                adj_matrix_bak = self.edge2adj(adj_matrix_bak, self.args.parallel_sampling, server_num, user_num, device)
                nodes = nodes.reshape(self.args.parallel_sampling, -1)

            if self.diffusion_type == 'hybrid' or self.diffusion_type == 'gaussian':
                pred_adj_ws = xt[:, -1, :, :]
                pred_adj_ws = torch.sigmoid(pred_adj_ws)
                # print(pred_adj_ws[0][4:, :4], gt_adj_ws[0][4:, :4])
            if not self.sparse and (self.diffusion_type == 'hybrid' or self.diffusion_type == 'categorical'):
                pred_adj_mat = np.where(xt[:, 0, :, :].float().cpu().detach().numpy() > self.cls_threshold, 1, 0)
                correct_cnt = np.all(np.equal(pred_adj_mat[:, 4:, :4], gt_adj_matrix.long().cpu().detach().numpy()[:, 4:, :4]), axis=-1)
                acc = np.mean(correct_cnt)
                # wrong_local, wrong_offload = count_rows(pred_adj_mat[:, 4:, :4], gt_adj_matrix.long().cpu().detach().numpy()[:, 4:, :4])
                # print(pred_adj_mat[0][4:, :4], gt_adj_matrix[0][4:, :4], acc, wrong_local / (gt_adj_matrix.shape[0] * 12), wrong_offload / (gt_adj_matrix.shape[0] * 12))
            if self.diffusion_type == 'hybrid':
                pred_cost, final_adj_mat, final_adj_ws = cost_calc(nodes, adj_matrix_bak, torch.where(xt[:, 0, :, :] > self.cls_threshold, 1, 0), pred_adj_ws,
                                                                   self.args.random_proprocess)
                min_idx = torch.argmin(pred_cost)
                if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
                    # dispose the bad solutions in this parallel sampling
                    solved_cost.append(pred_cost[min_idx].item())
                    solved_solutions.append(torch.cat((final_adj_mat[min_idx].unsqueeze(0), final_adj_ws[min_idx].unsqueeze(0)), dim=0))
                else:
                    exceed_ratio = torch.mean(pred_cost / gt_cost)

            if self.sparse:
                adj_matrix_bak = edge_index_bak
                nodes = nodes.reshape(-1)
        ###### raw denoising #######
        ############################
        if self.diffusion_type == 'hybrid' or self.diffusion_type == 'categorical':
            self.test_metrics["test/selection_acc"].append(acc)
        if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
            min_idx = torch.argmin(torch.tensor(solved_cost))
            self.best_solved_costs.append(solved_cost[min_idx])
            self.best_solved_solutions.append(solved_solutions[min_idx])
            exceed_ratio = torch.mean(solved_cost[min_idx] / gt_cost)
        self.test_metrics["test/exceed_ratio"].append(exceed_ratio.float().cpu().detach())
        # print(exceed_ratio)

        if self.args.do_train:
            self.log("test/exceed_ratio", exceed_ratio, sync_dist=True)
        else:
            self.log("ti/exceed_ratio", exceed_ratio, on_step=True, sync_dist=True)
        return self.test_metrics

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)
