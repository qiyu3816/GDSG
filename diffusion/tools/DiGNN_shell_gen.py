"""DiGNN running scripts auto gen"""
import glob

default_dataset_dir = '/root/GDSG/data/msmu-co'
train_datasets = {'3s6u': '3s6u_60000samples_20240606212245',
                  '3s8u': '3s8u_60000samples_20240606133850',
                  '4s10u': '4s10u_80000samples_20240605234002',
                  '4s12u': '4s12u_80000samples_20240509210626',
                  '7s24u': '7s24u_80000samples_20240606101948',
                  '7s27u': '7s27u_80000samples_20240606174323',
                  '10s31u': '10s31u_80000samples_20240607171012',
                  '10s36u': '10s36u_80000samples_20240607025117',
                  '20s61u': '20s61u_80000samples_20240605093253',
                  '20s68u': '20s68u_80000samples_20240606121707',
                  'gt3s6u': 'train_3s6u_2000samples',
                  'gt3s8u': 'train_3s8u_2000samples',
                  'gt4s10u': 'train_4s10u_2000samples',
                  'gt4s12u': 'train_4s12u_2000samples'}
alternate_server_scale = 4
alternate_dataset = '4s10u_80000samples_20240605234002'
alternate_step = 4
test_datasets = {'3s6u': '3s6u_2000samples_20240606182344',
                 '3s8u': '3s8u_2000samples_20240606181517',
                 '4s10u': '4s10u_1000samples_20240611134851',
                 '4s12u': '4s12u_1000samples_20240428223926',
                 '7s24u': '7s24u_2000samples_20240606104803',
                 '7s27u': '7s27u_2000samples_20240606181909',
                 '10s31u': '10s31u_2000samples_20240606183437',
                 '10s36u': '10s36u_2000samples_20240607103816',
                 '20s61u': '20s61u_2000samples_20240604223706',
                 '20s68u': '20s68u_2000samples_20240606123620',
                 'gt3s6u': 'test_3s6u_1000samples',
                 'gt3s8u': 'test_3s8u_1000samples',
                 'gt4s10u': 'test_4s10u_1000samples',
                 'gt4s12u': 'test_4s12u_1000samples',
                 'gt7s24u': 'refine200_7s24u',
                 'gt7s27u': 'refine200_7s27u',
                 'gt10s31u': 'refine200_10s31u',
                 'gt10s36u': 'refine200_10s36u',
                 'gt20s61u': 'refine100_20s61u',
                 'gt20s68u': 'refine100_20s68u'}

default_ckpt_dir = '/root/GDSG/data/msmu-co/models/'
ckpt_paths = {'3s6u': 'train_gnn_dense_3s6u20240629170635/vlvyu451/checkpoints/last.ckpt',
              '3s8u': 'train_gnn_dense_3s8u20240629170641/tz0mf4cs/checkpoints/last.ckpt',
              '4s10u': 'train_gnn_dense_4s10u20240629170646/qg9y6cvh/checkpoints/last.ckpt',
              '4s12u': 'train_gnn_dense_4s12u20240701220144/owalvp7z/checkpoints/last.ckpt',
              '7s24u': 'train_gnn_dense_7s24u20240629191156/2ah2db55/checkpoints/last.ckpt',
              '7s27u': 'train_gnn_dense_7s27u20240629195829/l26fpwiw/checkpoints/last.ckpt',
              '10s31u': 'train_gnn_dense_10s31u20240629204938/c7513tgl/checkpoints/last.ckpt',
              '10s36u': 'train_gnn_dense_10s36u20240629212310/lgfdnxf2/checkpoints/last.ckpt',
              '20s61u': 'train_gnn_sparse_20s61u20240629231600/ubhfvpza/checkpoints/last.ckpt',
              '20s68u': 'train_gnn_sparse_20s68u20240630021535/g6o3e75c/checkpoints/last.ckpt',
              'gt3s6u': 'train_gnn_dense_gt3s6u20240629174132/pi36i5un/checkpoints/last.ckpt',
              'gt3s8u': 'train_gnn_dense_gt3s8u20240629175514/xe8io40m/checkpoints/last.ckpt',
              'gt4s10u': 'train_gnn_dense_gt4s10u20240629184326/3qiw4tqw/checkpoints/last.ckpt',
              'gt4s12u': 'train_gnn_dense_gt4s12u20240629204518/ww10dxsq/checkpoints/last.ckpt'}
graph_modes = {'3s6u': 'dense',
               '3s8u': 'dense',
               '4s10u': 'dense',
               '4s12u': 'dense',
               '7s24u': 'dense',
               '7s27u': 'dense',
               '10s31u': 'dense',
               '10s36u': 'dense',
               '20s61u': 'sparse',
               '20s68u': 'sparse',
               'gt3s6u': 'dense',
               'gt3s8u': 'dense',
               'gt4s10u': 'dense',
               'gt4s12u': 'dense'}
# B_e_D_i_H_L_p
# In DiGNN, only B_e_H_L are used.
model_settings = {'3s6u': '256_50_200_5_256_5_16',
                  '3s8u': '256_50_200_5_256_5_16',
                  '4s10u': '256_50_200_5_256_5_8',
                  '4s12u': '256_50_200_5_256_5_8',
                  '7s24u': '256_30_100_5_256_5_8',
                  '7s27u': '128_30_100_5_256_5_8',
                  '10s31u': '64_20_100_5_256_6_8',
                  '10s36u': '64_20_100_5_256_6_8',
                  '20s61u': '128_30_100_5_256_8_8',
                  '20s68u': '128_30_100_5_256_8_8',
                  'gt3s6u': '32_100_200_5_256_8_16',
                  'gt3s8u': '32_100_200_5_256_8_16',
                  'gt4s10u': '32_100_200_5_256_8_16',
                  'gt4s12u': '32_100_200_5_256_8_16'}

def preface():
    cur_script = 'export PYTHONPATH=\"$PWD:$PYTHONPATH\"\n'
    cur_script += 'export CUDA_VISIBLE_DEVICES=0\n'
    cur_script += '\n# shellcheck disable=SC2155\n'
    cur_script += 'export WANDB_RUN_ID=$(python -c \"import wandb; print(wandb.util.generate_id())\")\n'
    cur_script += 'echo \"WANDB_ID is $WANDB_RUN_ID\"\n'
    cur_script += '\npython -u gnn_run.py \\\n'
    cur_script += '  --task \"msco\" \\\n'
    cur_script += '  --project_name \"official_infocom25\" \\\n'
    return cur_script

def train_logger_name(graph_mode, server_user, alternate_train):
    alternate_suffix = '_alt' if alternate_train else ''
    return f"train_DiGNN_{graph_mode}_{server_user}{alternate_suffix}"

def test_logger_name(graph_mode, server_user, cur_server_user, random_proprocess):
    random_proprocess_suffix = '_rand' if random_proprocess else ''
    if cur_server_user == server_user:
        return f"test_DiGNN_{graph_mode}_{server_user}{random_proprocess_suffix}"
    else:
        return f"test_DiGNN_{graph_mode}_{server_user}_cross_{cur_server_user}{random_proprocess_suffix}"

def default_settings():
    cur_script = '  --learning_rate 0.0002 \\\n'
    cur_script += '  --weight_decay 0.0001 \\\n'
    cur_script += '  --lr_scheduler \"cosine-decay\" \\\n'
    cur_script += '  --validation_examples 8 \\\n'
    return cur_script

def dataset_splits(server_user, alternate_train=False, cur_server_user=None):
    server_num = int(server_user.split('s')[0][2:]) if server_user.startswith('gt') else int(server_user.split('s')[0])

    cur_script = f'  --training_split \"{server_num}server/{train_datasets[server_user]}.txt\" \\\n'
    cur_script += f'  --validation_split \"{server_num}server/{test_datasets[server_user]}.txt\" \\\n'
    if cur_server_user is not None:
        cur_server_num = int(cur_server_user.split('s')[0][2:]) if cur_server_user.startswith('gt') else int(cur_server_user.split('s')[0])
        cur_script += f'  --test_split \"{cur_server_num}server/{test_datasets[cur_server_user]}.txt\" \\\n'
    else:
        cur_script += f'  --test_split \"{server_num}server/{test_datasets[server_user]}.txt\" \\\n'
    if alternate_train is True:
        assert server_num > alternate_server_scale
        cur_script += f'  --alternate_split \"{alternate_server_scale}server/{alternate_dataset}.txt\" \\\n'
        cur_script += f'  --alternate_step {alternate_step} \\\n'
    return cur_script

def model_setting_script(work_type, server_user):
    setting = model_settings[server_user]
    batch_size = int(setting.split('_')[0])
    num_epochs = int(setting.split('_')[1])
    diffusion_steps = int(setting.split('_')[2])
    inference_diffusion_steps = int(setting.split('_')[3])
    hidden_dim = int(setting.split('_')[4])
    n_layers = int(setting.split('_')[5])
    parallel_sampling = int(setting.split('_')[6])

    cur_ckpt = default_ckpt_dir + ckpt_paths[server_user]

    cur_script = f'  --batch_size {batch_size} \\\n'
    cur_script += f'  --num_epochs {num_epochs} \\\n'
    cur_script += f'  --hidden_dim {hidden_dim} \\\n'
    cur_script += f'  --n_layers {n_layers}'
    if work_type == 'test':
        cur_script += f' \\\n  --ckpt_path \"{cur_ckpt}\"'
    return cur_script

def train_script(server_user, alternate_train):
    graph_mode = graph_modes[server_user]

    cur_script = preface()
    cur_script += f'  --wandb_logger_name \"{train_logger_name(graph_mode, server_user, alternate_train)}\" \\\n'
    cur_script += '  --do_train \\\n'
    cur_script += default_settings()
    cur_script += f'  --storage_path \"{default_dataset_dir}\" \\\n'
    cur_script += dataset_splits(server_user, alternate_train)
    cur_script += model_setting_script('train', server_user)
    if graph_mode == 'sparse':
        cur_script += ' \\\n  --sparse'  # model_setting_script在末尾没有双斜杠 这里要加sparse的话必须补上

    train_shell_file_name = f"train_DiGNN_{server_user}_{graph_mode}.sh"
    return cur_script, train_shell_file_name

def test_script(server_user, cur_server_user, random_proprocess):
    graph_mode = graph_modes[server_user]
    random_proprocess_suffix = '_rand' if random_proprocess else ''

    cur_script = preface()
    cur_script += f'  --wandb_logger_name \"{test_logger_name(graph_mode, server_user, cur_server_user, random_proprocess)}\" \\\n'
    cur_script += '  --do_test \\\n'
    cur_script += default_settings()
    cur_script += f'  --storage_path \"{default_dataset_dir}\" \\\n'
    cur_script += dataset_splits(server_user, cur_server_user=cur_server_user)
    cur_script += model_setting_script('test', server_user)
    if graph_mode == 'sparse':
        cur_script += ' \\\n  --sparse'  # model_setting_script在末尾没有双斜杠 这里要加sparse的话必须补上
    if random_proprocess:
        cur_script += ' \\\n  --random_proprocess'

    if server_user == cur_server_user:
        test_shell_file_name = f"test_DiGNN_{server_user}{random_proprocess_suffix}.sh"
    else:
        test_shell_file_name = f"test_DiGNN_{server_user}_cross_{cur_server_user}{random_proprocess_suffix}.sh"
    return cur_script, test_shell_file_name

def delete_xftp_slash_r():
    """
    Use it in the shells' dir on linux.
    """
    sh_files = glob.glob('*.sh')

    for sh_file in sh_files:
        with open(sh_file, 'r') as f:
            lines = f.readlines()
        with open(sh_file, 'w') as f:
            for line in lines:
                f.write(line.replace('\r', ''))


if __name__ == '__main__':
    work_type = 'test'
    server_user = '20s68u'
    alternate_train = False
    random_proprocess = False

    if work_type == 'train':
        cur_script, shell_file_name = train_script(server_user, alternate_train)
        with open(shell_file_name, 'w') as f:
            f.write(cur_script)
        f.close()
    elif work_type == 'test':
        random_proprocess_suffix = '_rand' if random_proprocess else ''

        ensemble_shell = ''
        cur_server_user_s = list(test_datasets.keys())
        for cur_server_user in cur_server_user_s:
            cur_script, shell_file_name = test_script(server_user, cur_server_user, random_proprocess)
            with open(shell_file_name, 'w') as f:
                f.write(cur_script)
            f.close()
            ensemble_shell += f"./{shell_file_name} \n"
            ensemble_shell += "sleep 5 \n"
        with open(f"ensemble_test_{server_user}{random_proprocess_suffix}.sh", 'w') as f:
            f.write(ensemble_shell)
        f.close()
    elif work_type == 'win':
        delete_xftp_slash_r()
