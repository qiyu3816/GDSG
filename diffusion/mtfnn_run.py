import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader as GeoDataLoader
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from models.MTFNN import MTFNN
from co_datasets.msco_graph_dataset import MSCOGraphDataset
from utils.msco_utils import cost_calc, calc_cost_upper


default_dataset_dir = 'D:/Codes/DT4SG/data/msmu-co'
train_datasets = {'3s6u': '3server/3s6u_60000samples_20240606212245',
                  '3s8u': '3server/3s8u_60000samples_20240606133850',
                  '4s10u': '4server/4s10u_80000samples_20240605234002',
                  '4s12u': '4server/4s12u_80000samples_20240509210626',
                  '7s24u': '7server/7s24u_80000samples_20240606101948',
                  '7s27u': '7server/7s27u_80000samples_20240606174323',
                  '10s31u': '10server/10s31u_80000samples_20240607171012',
                  '10s36u': '10server/10s36u_80000samples_20240607025117',
                  '20s61u': '20server/20s61u_80000samples_20240605093253',
                  '20s68u': '20server/20s68u_80000samples_20240606121707',
                  'gt3s6u': '3server/train_3s6u_2000samples',
                  'gt3s8u': '3server/train_3s8u_2000samples',
                  'gt4s10u': '4server/train_4s10u_2000samples',
                  'gt4s12u': '4server/train_4s12u_2000samples'}
test_datasets = {'3s6u': '3server/3s6u_2000samples_20240606182344',
                 '3s8u': '3server/3s8u_2000samples_20240606181517',
                 '4s10u': '4server/4s10u_1000samples_20240611134851',
                 '4s12u': '4server/4s12u_1000samples_20240428223926',
                 '7s24u': '7server/7s24u_2000samples_20240606104803',
                 '7s27u': '7server/7s27u_2000samples_20240606181909',
                 '10s31u': '10server/10s31u_2000samples_20240606183437',
                 '10s36u': '10server/10s36u_2000samples_20240607103816',
                 '20s61u': '20server/20s61u_2000samples_20240604223706',
                 '20s68u': '20server/20s68u_2000samples_20240606123620',
                 'gt3s6u': '3server/test_3s6u_1000samples',
                 'gt3s8u': '3server/test_3s8u_1000samples',
                 'gt4s10u': '4server/test_4s10u_1000samples',
                 'gt4s12u': '4server/test_4s12u_1000samples',
                 'gt7s24u': '7server/refine200_7s24u',
                 'gt7s27u': '7server/refine200_7s27u',
                 'gt10s31u': '10server/refine200_10s31u',
                 'gt10s36u': '10server/refine200_10s36u',
                 'gt20s61u': '20server/refine100_20s61u',
                 'gt20s68u': '20server/refine100_20s68u'}
# B, E, hidden_dim, layer_num
model_settings = {'3s6u': '256_50_512_5',
                  '3s8u': '256_50_512_5',
                  '4s10u': '256_50_512_5',
                  '4s12u': '256_50_512_5',
                  '7s24u': '256_50_512_5',
                  '7s27u': '256_50_512_5',
                  '10s31u': '256_50_1024_6',
                  '10s36u': '256_50_1024_6',
                  '20s61u': '256_50_1024_8',
                  '20s68u': '256_50_1024_8',
                  'gt3s6u': '32_200_512_5',
                  'gt3s8u': '32_200_512_5',
                  'gt4s10u': '32_200_512_5',
                  'gt4s12u': '32_200_512_5'}

default_ckpt_dir = 'D:/Codes/DT4SG/data/msmu-co/models/'
ckpt_paths = {'3s6u': 'mtfnn_3s6u.pth',
              '3s8u': 'mtfnn_3s8u.pth',
              '4s10u': 'mtfnn_4s10u.pth',
              '4s12u': 'mtfnn_4s12u.pth',
              '7s24u': 'mtfnn_7s24u.pth',
              '7s27u': 'mtfnn_7s27u.pth',
              '10s31u': 'mtfnn_10s31u.pth',
              '10s36u': 'mtfnn_10s36u.pth',
              '20s61u': 'mtfnn_20s61u.pth',
              '20s68u': 'mtfnn_20s68u.pth',
              'gt3s6u': 'mtfnn_gt3s6u.pth',
              'gt3s8u': 'mtfnn_gt3s8u.pth',
              'gt4s10u': 'mtfnn_gt4s10u.pth',
              'gt4s12u': 'mtfnn_gt4s12u.pth'}

class ModelRunner:
    def __init__(self, model, train_dataset, test_dataset, device):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        self.cost_upper = None
        self.cost_lower = 0
        self.cls_threshold = 0.9
        self.test_metrics = None

    def normalize_input(self, adj_matrix, server_num, user_num):
        self.cost_upper = calc_cost_upper(server_num, user_num)
        adj_matrix[:, 1:4, :, :] = ((adj_matrix[:, 1:4, :, :] - self.cost_lower) / (self.cost_upper - self.cost_lower)).clamp(0, 1)

    def train(self, epochs, batch_size):
        self.model.train()
        dataloader = GeoDataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Training epoch {epoch+1}/{epochs}"):
                _, nodes, adj_matrix, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
                server_num = self.model.server_num
                user_num = self.model.user_num
                self.normalize_input(adj_matrix, server_num, user_num)

                adj_matrix = adj_matrix.to(self.device)
                batch_size = adj_matrix.shape[0]
                input_adj = adj_matrix[:, :, server_num:, :server_num].permute(0, 2, 3, 1).reshape(batch_size, -1)
                cls_out, reg_out = self.model(input_adj)

                # Compute loss
                loss_cls = nn.CrossEntropyLoss()
                loss_reg = F.mse_loss
                cls_out = cls_out.reshape(-1, 2)
                gt_adj_matrix = gt_adj_matrix.to(self.device)
                gt_adj_ws = gt_adj_ws.to(self.device)
                cls_loss = loss_cls(cls_out, gt_adj_matrix[:, server_num:, :server_num].reshape(-1).long())
                reg_loss = loss_reg(reg_out, gt_adj_ws[:, server_num:, :server_num].reshape(batch_size, -1))
                loss = cls_loss + reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self):
        dataloader = GeoDataLoader(self.test_dataset, batch_size=16, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing"):
                _, nodes, adj_matrix, local_costs, gt_adj_matrix, gt_adj_ws, gt_cost = batch
                server_num = self.model.server_num
                user_num = self.model.user_num
                adj_matrix_bak = adj_matrix.clone()
                self.normalize_input(adj_matrix, server_num, user_num)

                adj_matrix = adj_matrix.to(self.device)
                batch_size = adj_matrix.shape[0]
                input_adj = adj_matrix[:, :, server_num:, :server_num].permute(0, 2, 3, 1).reshape(batch_size, -1)
                cls_out, y_pred_reg = self.model(input_adj)
                y_pred_cls = torch.where(cls_out.reshape(-1, 2)[:, 1] > self.cls_threshold, 0, 1)
                y_pred_cls = y_pred_cls.reshape(batch_size, user_num, server_num)
                y_pred_reg = y_pred_reg.reshape(batch_size, user_num, server_num)

                padding_y_pred_cls = torch.zeros(batch_size, server_num + user_num, server_num + user_num)
                padding_y_pred_reg = torch.zeros(batch_size, server_num + user_num, server_num + user_num)
                padding_y_pred_cls[:, server_num:, :server_num] = y_pred_cls
                padding_y_pred_reg[:, server_num:, :server_num] = y_pred_reg

                pred_cost, final_adj_mat, final_adj_ws = cost_calc(nodes, adj_matrix_bak, padding_y_pred_cls, padding_y_pred_reg)
                exceed_ratios = pred_cost / gt_cost.squeeze(-1)
                if self.test_metrics is None:
                    self.test_metrics = {"test/exceed_ratio": []}
                self.test_metrics["test/exceed_ratio"] += exceed_ratios.tolist()


def mtfnn_run():
    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_settings = train_datasets.keys()
    for setting in exp_settings:
        print(f"Train {setting}   ####################################################################################")
        if setting.startswith('gt'):
            server_num = int(setting[2:].split('s')[0])
        else:
            server_num = int(setting.split('s')[0])
        user_num = int(setting.split('s')[-1][:-1])
        batch_size = int(model_settings[setting].split('_')[0])
        epochs = int(model_settings[setting].split('_')[1])
        hidden_dim = int(model_settings[setting].split('_')[2])
        layer_num = int(model_settings[setting].split('_')[3])

        model = MTFNN(server_num=server_num, user_num=user_num, hidden_dim=hidden_dim, layer_num=layer_num)
        train_dataset = MSCOGraphDataset(data_file=f'{default_dataset_dir}/{train_datasets[setting]}.txt')
        test_dataset = MSCOGraphDataset(data_file=f'{default_dataset_dir}/{test_datasets[setting]}.txt')

        runner = ModelRunner(model, train_dataset, test_dataset, device)
        runner.train(epochs=epochs, batch_size=batch_size)
        torch.save(model.state_dict(), f'{default_dataset_dir}/models/mtfnn_{setting}.pth')
        print(f"Train {setting} model stored at {default_dataset_dir}/models/mtfnn_{setting}.pth")
        runner.test()
        cur_res = np.mean(runner.test_metrics["test/exceed_ratio"])
        print(f"Test results for setting {setting}: {cur_res}   ############")

def mtfnn_read_test():
    if torch.cuda.is_available():
        print(f"Found cuda. Device count: {torch.cuda.device_count()}, the 0 is {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_settings = list(train_datasets.keys())

    overall_res = []
    for setting in exp_settings:
        if setting.startswith('gt'):
            server_num = int(setting[2:].split('s')[0])
        else:
            server_num = int(setting.split('s')[0])
        user_num = int(setting.split('s')[-1][:-1])
        hidden_dim = int(model_settings[setting].split('_')[2])
        layer_num = int(model_settings[setting].split('_')[3])

        model = MTFNN(server_num=server_num, user_num=user_num, hidden_dim=hidden_dim, layer_num=layer_num)
        model.load_state_dict(torch.load(f"{default_ckpt_dir}/{ckpt_paths[setting]}"))

        train_dataset = MSCOGraphDataset(data_file=f'{default_dataset_dir}/{train_datasets[setting]}.txt')
        test_sets = test_datasets.keys()
        cur_round_res = []
        pure_model_setting = setting[2:] if setting.startswith('gt') else setting
        for test_set in test_sets:
            if not test_set.endswith(pure_model_setting):
                cur_round_res.append(-1)
                continue
            test_dataset = MSCOGraphDataset(data_file=f'{default_dataset_dir}/{test_datasets[test_set]}.txt')
            runner = ModelRunner(model, train_dataset, test_dataset, device)
            runner.test()
            cur_res = np.mean(runner.test_metrics["test/exceed_ratio"])
            cur_round_res.append(cur_res)
        overall_res.append(cur_round_res)

    df = pd.DataFrame(overall_res, index=exp_settings, columns=test_datasets.keys())
    df.to_csv('MTFNN_all_res.csv')
    print(f"Test results stored as MTFNN_all_res.csv")


if __name__ == '__main__':
    # mtfnn_run()
    mtfnn_read_test()
