"""A meta PyTorch Lightning model for training and evaluating DIFUSCO models."""
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info

from utils.lr_schedulers import get_schedule_fn
from models.gnn_di import GNNDi
from co_datasets.msco_graph_dataset import MSCOGraphDataset
from utils.msco_utils import cost_calc, validity_deal, calc_cost_upper


class GNNDiModel(pl.LightningModule):
    def __init__(self,
                 param_args,
                 node_feature_only=False):
        super(GNNDiModel, self).__init__()
        self.args = param_args
        self.sparse = self.args.sparse or node_feature_only
        self.frozen = False
        out_channels = 3

        self.model = GNNDi(
            n_layers=self.args.n_layers,
            hidden_dim=self.args.hidden_dim,
            out_channels=out_channels,
            aggregation=self.args.aggregation,
            sparse=self.sparse,
            use_activation_checkpoint=self.args.use_activation_checkpoint,
            node_feature_only=node_feature_only,
        )
        self.num_training_steps_cached = None

        self.train_dataset = MSCOGraphDataset(data_file=os.path.join(self.args.storage_path, self.args.training_split), sparse=self.sparse)
        self.test_dataset = MSCOGraphDataset(data_file=os.path.join(self.args.storage_path, self.args.test_split), sparse=self.sparse)
        self.validation_dataset = MSCOGraphDataset(data_file=os.path.join(self.args.storage_path, self.args.validation_split), sparse=self.sparse)

        self.cost_upper = None
        self.cost_lower = 0

        self.cls_threshold = 0.9

        self.test_metrics = None

    def forward(self, nodes, edges):
        return self.model(nodes, edges)

    def normalize_input(self, adj_matrix, server_num, user_num):
        self.cost_upper = calc_cost_upper(server_num, user_num)
        if self.sparse:
            adj_matrix[3:6, :] = ((adj_matrix[3:6, :] - self.cost_lower) / (self.cost_upper - self.cost_lower)).clamp(0, 1)
        else:
            adj_matrix[:, 1:4, :, :] = ((adj_matrix[:, 1:4, :, :] - self.cost_lower) / (self.cost_upper - self.cost_lower)).clamp(0, 1)

    def training_step(self, batch, batch_idx):
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

        y_pred = self.forward(
            nodes.float().to(gt_adj_matrix.device),
            adj_matrix.float().to(gt_adj_matrix.device),
        )
        if self.sparse:
            x0_pred_cls = y_pred[:, :2]
            x0_pred_reg = y_pred[:, -1]
            gt_adj_matrix = gt_adj_matrix.reshape(-1)
            gt_adj_ws = gt_adj_ws.reshape(-1)
        else:
            x0_pred_cls = y_pred[:, :2, :, :]
            x0_pred_reg = y_pred[:, -1, :, :]

        loss_cls = nn.CrossEntropyLoss()
        loss_reg = F.mse_loss

        if self.args.grad_calculate:
            loss_cls_val = loss_cls(x0_pred_cls, gt_adj_matrix.long())
            loss_reg_val = loss_reg(x0_pred_reg, gt_adj_matrix.long().float())
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
            with open(f'GNN_{server_num}s{user_num}u_grads.txt', 'a') as f:
                f.write(f'Epoch: {self.current_epoch}\n')
                for name in grad_reg:
                    f.write(
                        f'Parameter: {name}, \nClassification loss gradient: \n{grad_cls[name]}, \nRegression loss gradient: \n{grad_reg[name]}\n\n')

            loss = loss_cls(x0_pred_cls, gt_adj_matrix.long())
            self.log("train/cls_loss", loss)
            if self.frozen or self.args.freeze_epoch <= 0:
                reg_loss = loss_reg(x0_pred_reg, gt_adj_matrix.long().float())
                loss += reg_loss
                self.log("train/reg_loss", reg_loss)
        else:
            loss = loss_cls(x0_pred_cls, gt_adj_matrix.long())
            self.log("train/cls_loss", loss)
            if self.frozen or self.args.freeze_epoch <= 0:
                reg_loss = loss_reg(x0_pred_reg, gt_adj_ws.float())
                loss += reg_loss
                self.log("train/reg_loss", reg_loss)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
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

        y_pred = self.forward(
            nodes.float().to(gt_adj_matrix.device),
            adj_matrix.float().to(gt_adj_matrix.device),
        )
        y_pred_cls = torch.where(y_pred[:, 1] > self.cls_threshold, 1, 0)
        y_pred_reg = y_pred[:, -1]
        if self.sparse:
            nodes = nodes.reshape(batch_size, -1)
            edge_index = self.deduplicate_edge_index(edge_index, batch_size)
            adj_matrix_bak[:2, :] = edge_index
            adj_matrix_bak = self.edge2adj(adj_matrix_bak, batch_size, server_num, user_num, device)
            yt = torch.cat((y_pred_cls.unsqueeze(-1), y_pred_reg.unsqueeze(-1)), dim=1)
            yt = self.pred_edge2adj(yt, edge_index, batch_size, server_num, user_num, device)
            y_pred_cls = yt[:, 0]
            y_pred_reg = yt[:, 1]

        if self.test_metrics is None:
            self.test_metrics = {"test/exceed_ratio": []}
        pred_cost, final_adj_mat, final_adj_ws = cost_calc(nodes, adj_matrix_bak, y_pred_cls, y_pred_reg,
                                                           self.args.random_proprocess)
        exceed_ratio = torch.mean(pred_cost / gt_cost)
        exceed_ratios = pred_cost / gt_cost.squeeze(-1)
        self.test_metrics["test/exceed_ratio"] += exceed_ratios.tolist()

        if self.args.do_train:
            self.log("test/exceed_ratio", exceed_ratio, sync_dist=True)
        else:
            self.log("ti/exceed_ratio", exceed_ratio, on_step=True, sync_dist=True)
        return self.test_metrics

    def on_test_epoch_end(self):
        merged_metrics = {}
        for k, v in self.test_metrics.items():
            merged_metrics[k] = float(np.mean(v))

        self.logger.log_metrics(merged_metrics, step=self.global_step)

    def get_total_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches * len(dataset)
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        return self.num_training_steps_cached

    def configure_optimizers(self):
        rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
        rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

        if self.args.lr_scheduler == "constant":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def duplicate_edge_index(self, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, self.args.parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def deduplicate_edge_index(self, edge_index, batch_size):
        """
        Only for msco.
        """
        edge_num = edge_index.shape[1] // batch_size
        for i in range(batch_size):
            if i == 0:
                continue
            edge_index[:, i * edge_num:(i + 1) * edge_num] = edge_index[:, :edge_num]
        return edge_index

    def edge2adj(self, edge_index, batch_size, server_num, user_num, device):
        """
        Only for msco GNNDi.
        """
        adj = torch.zeros((batch_size, edge_index.shape[0] - 2, server_num + user_num, server_num + user_num), device=device)
        edge_num = server_num * user_num
        edge_index_long = edge_index[:3].long()
        for i in range(batch_size):
            for j in range(edge_num):
                if edge_index_long[2, i * edge_num + j] == 1:
                    adj[i, :, edge_index_long[0, i * edge_num + j], edge_index_long[1, i * edge_num + j]] = edge_index[2:, i * edge_num + j]
        return adj

    def pred_edge2adj(self, xt, edge_index, batch_size, server_num, user_num, device):
        """
        Only for msco.
        """
        adj = torch.zeros((batch_size, 2, server_num + user_num, server_num + user_num), device=device)
        edge_num = server_num * user_num
        edge_index_long = edge_index.long()
        for i in range(batch_size):
            for j in range(edge_num):
                adj[i, 0, edge_index_long[0, i * edge_num + j], edge_index_long[1, i * edge_num + j]] = xt[i * edge_num + j, 0]
                adj[i, 1, edge_index_long[0, i * edge_num + j], edge_index_long[1, i * edge_num + j]] = xt[i * edge_num + j, 1]
        return adj


    def train_dataloader(self):
        batch_size = self.args.batch_size
        train_dataloader = GraphDataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
            persistent_workers=True, drop_last=True)
        return train_dataloader

    def test_dataloader(self):
        batch_size = self.args.test_batch
        print("Test dataset size:", len(self.test_dataset))
        test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return test_dataloader

    def val_dataloader(self):
        batch_size = 1
        val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
        print("Validation dataset size:", len(val_dataset))
        val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return val_dataloader
