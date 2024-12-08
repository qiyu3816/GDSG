import os
import shutil
import re
import torch
import math
import numpy as np
import ast

def param_avg_across_all_epoch(param_avg_angle_path):
    # for all parameter, calculate the average value and standard deviation
    # of the angle across all epoches
    in_this_range = 0
    total = 0
    total_val = 0
    total_diff = 0
    for angle_f in os.listdir(param_avg_angle_path):
        if not angle_f == 'param_avg_angle.npy' and not angle_f == 'param_to_index.txt':
            with open(os.path.join(param_avg_angle_path, angle_f), 'r') as f:
                angle = float(f.readline())
                if angle == 0.0:
                    print('Zero angle file still exist: ', angle_f)
                total = total + 1
                total_val = total_val + angle
                if 100.0 > angle > 80.0:
                    in_this_range = in_this_range + 1

    avg_val = total_val / total
    for angle_f in os.listdir(param_avg_angle_path):
        if not angle_f == 'param_avg_angle.npy' and not angle_f == 'param_to_index.txt':
            with open(os.path.join(param_avg_angle_path, angle_f), 'r') as f:
                angle = float(f.readline())
                total_diff = total_diff + (angle - avg_val) ** 2

    std_diff = (total_diff / total) ** 0.5

    print('Total: ', total)
    print('In range 80° ~ 100°: ', in_this_range)
    print('Percentage: ', in_this_range / total * 100.0, '%')
    print('Average: ', total_val / total)
    print('Standard deviation: ', std_diff)

def delete_zero_angle(param_angle_file):
    #
    if not os.path.exists(param_angle_file):
        print('Parameter angle directory does not exist.')
        exit(1)
    else:
        print('Deleting zero angle files.')
        print(f'Parameter angle files are stored in : {param_angle_file}')

    for param_angle_f in os.listdir(param_angle_file):
        if not param_angle_f == 'param_avg_angle.npy' and not param_angle_f == 'param_to_index.txt':
            with open(os.path.join(param_angle_file, param_angle_f), 'r') as f:
                remove_flag = 0
                angle = float(f.read().split('\n')[0])
                if angle == 0.0:
                    remove_flag = 1
            if remove_flag:
                os.remove(os.path.join(param_angle_file, param_angle_f))
    print('Zero angle files are deleted.')

def avg_angle_for_param(avg_angle_files_path, avg_angle_param_path):
    if not os.path.exists(avg_angle_files_path):
        print(f'Path {avg_angle_files_path} does not exist')
        exit(1)

    # Mapping: parameter name <--> index
    param_to_index = {}
    index_to_param = {}

    # The average angle for each parameter
    param_avg_angle = []

    for epoch_dir in os.listdir(avg_angle_files_path):
        epoch = int(epoch_dir.split('epoch')[1])
        for param_file in os.listdir(os.path.join(avg_angle_files_path, epoch_dir)):
            param = param_file.split('.txt')[0]
            for ch in param:
                if ch == '.':
                    param = param.replace(ch, '_')

            # if a new parameter
            if param not in param_to_index:
                param_to_index[param] = len(param_to_index)
                index_to_param[len(index_to_param)] = param
                param_avg_angle.append([])

            with open(os.path.join(avg_angle_files_path, epoch_dir, param_file), 'r') as f:
                avg_angle = float(f.read())
                param_avg_angle[param_to_index[param]].append(avg_angle)

    param_avg_angle = np.array(param_avg_angle)

    if not os.path.exists(avg_angle_param_path):
        os.makedirs(avg_angle_param_path)

    np.save(os.path.join(avg_angle_param_path, 'param_avg_angle.npy'), param_avg_angle)
    with open(os.path.join(avg_angle_param_path, 'param_to_index.txt'), 'w') as f:
        for param, index in param_to_index.items():
            f.write(f'{param}: {index}\n')
    param_avg_angle_mean = np.mean(param_avg_angle, axis=1)
    for param, index in param_to_index.items():
        with open(os.path.join(avg_angle_param_path, f'{param}.txt'), 'w') as f:
            f.write(str(param_avg_angle_mean[index]) + '\n')


def avg_angle_calculation(angle_files_path, avg_angle_files_path):
    # calculate the average angle of each parameter across all epochs
    if not os.path.exists(angle_files_path):
        print('Angle_files directory does not exist.')
        exit(1)
    else:
        print('Average angle calculation begins.')
        print(f'Average angle files will be stored in : {avg_angle_files_path}')
        if not os.path.exists(avg_angle_files_path):
            os.makedirs(avg_angle_files_path)

    for epoch_dir in os.listdir(angle_files_path):
        epoch = int(epoch_dir.split('epoch')[1])
        if not os.path.exists(os.path.join(avg_angle_files_path, f'avg_angle_epoch{epoch}')):
            os.makedirs(os.path.join(avg_angle_files_path, f'avg_angle_epoch{epoch}'))
        for param_file in os.listdir(os.path.join(angle_files_path, epoch_dir)):
            param = param_file.split('.txt')[0]
            with open(os.path.join(angle_files_path, epoch_dir, param_file), 'r') as f:
                lines = f.readlines()
                angles = []
                for line in lines:
                    angles.append(float(line))
                avg_angle = sum(angles) / len(angles)
                with open(os.path.join(avg_angle_files_path, f'avg_angle_epoch{epoch}', f'{param}.txt'), 'w') as f:
                    f.write(str(avg_angle))

    print('Average angle calculation is done.')

def calculate_angle(grad1, grad2):
    dot_product = torch.dot(grad1, grad2)
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)
    if norm1 == 0 or norm2 == 0:
        return torch.tensor(0.0)
    else:
        cos_theta = dot_product / (norm1 * norm2)
        angle_rad = torch.acos(cos_theta)
        angle_degree = angle_rad / math.pi * 180
        return angle_degree


def angle_calculation(grads_storage_path, angle_storage_path):
    # calculates the angle between the gradients of the classification and regression losses
    if not os.path.exists(grads_storage_path):
        print('Grads_files directory does not exist.')
        exit(1)
    else:
        print('Angle calculation begins.')
        print(f'Angle files will be stored in : {angle_storage_path}')
        if not os.path.exists(angle_storage_path):
            os.makedirs(angle_storage_path)

    for epoch_dir in os.listdir(grads_storage_path):
        epoch = int(epoch_dir.split('epoch')[1])
        if not os.path.exists(os.path.join(angle_storage_path, f'angle_epoch{epoch}')):
            os.makedirs(os.path.join(angle_storage_path, f'angle_epoch{epoch}'))
        for param_file in os.listdir(os.path.join(grads_storage_path, epoch_dir)):
            param = param_file.split('.txt')[0]
            with open(os.path.join(grads_storage_path, epoch_dir, param_file), 'r') as f:
                lines = f.readlines()
                for line in enumerate(lines):
                    if line[1].startswith('C'):
                        data = '[' + line[1].split('[')[1].split('device')[0].split(']')[0].strip() + ']'
                        grad_cls_data = torch.tensor(ast.literal_eval(data))
                    elif line[1].startswith('R'):
                        data = '[' + line[1].split('[')[1].split('device')[0].split(']')[0].strip() + ']'
                        grad_reg_data = torch.tensor(ast.literal_eval(data))
                        angle = calculate_angle(grad_cls_data, grad_reg_data)
                        with open(os.path.join(angle_storage_path, f'angle_epoch{epoch}', f'{param}.txt'), 'a') as angle_file:
                            angle_file.write(f'{angle}\n')

    print('Angle calculation is done.')

def grads_reconstruct(grads_path, storage_path):
    # Read the grads.txt file and divide it into several files to analyze
    print('Grad reconstruction begins.')
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    with open(grads_path, 'r') as f:
        epoch = None
        next_cls = False
        next_reg = False
        name = None
        for line in f:
            if next_cls:
                path = os.path.join(storage_path, f'epoch{epoch}', f'{name}.txt')
                with open(path, 'a') as grad_cls_file:
                    parts = line.split('\n')
                    grad_cls_file.write(parts[0])
            elif next_reg:
                path = os.path.join(storage_path, f'epoch{epoch}', f'{name}.txt')
                with open(path, 'a') as grad_reg_file:
                    parts = line.split('\n')
                    grad_reg_file.write(parts[0])

            if line.startswith('Epoch'):
                epoch = int(line.split(':')[1])
                if not os.path.exists(f'{storage_path}/epoch{epoch}'):
                    os.makedirs(f'{storage_path}/epoch{epoch}')
                next_cls = False
                next_reg = False
            elif line.startswith('Parameter'):
                parts = re.split(r', (?=[a-zA-Z])', line)  # split by comma followed by a space and a letter
                name = parts[0].split(':')[1].split(',')[0].strip()
                next_cls = False
                next_reg = False
            elif line.startswith('Classification loss gradient'):
                path = os.path.join(storage_path, f'epoch{epoch}', f'{name}.txt')
                with open(path, 'a') as grad_cls_file:
                    grad_cls_file.write('C')
                next_cls = True
                next_reg = False
            elif line.startswith('Regression loss gradient'):
                path = os.path.join(storage_path, f'epoch{epoch}', f'{name}.txt')
                with open(path, 'a') as grad_reg_file:
                    grad_reg_file.write('R')
                next_cls = False
                next_reg = True
            elif line.find('device') != -1:
                with open(os.path.join(storage_path, f'epoch{epoch}', f'{name}.txt'), 'a') as temp_file:
                    temp_file.write('\n')
                next_cls = False
                next_reg = False
    print('Grad reconstruction is done.')

def show_specific_angles(path):
    # choose the specific parameters you need to show
    for file in os.listdir(path):
            if 'node_embed' in file or 'edge_embed' in file:
                with open(os.path.join(path, file), 'r') as f:
                    angle = float(f.readline())
                    print(f'{file} : {angle}°\n')


if __name__ == '__main__':
    model = 'GDSG'
    dataset_is_gt = False
    sever_num = 7
    user_num = 27
    save_information = False

    if model == 'GDSG':
        path_split = 'gdsgGradResult'
    elif model == 'DiGNN':
        path_split = 'gnnGradResult'
    else:
        raise NotImplementedError
    original_grads_file_path = f'/root/CPY/DT4SG/difusco/{model}_{sever_num}s{user_num}u_grads.txt'
    if dataset_is_gt:
        result_storage_path = f'/root/CPY/DT4SG/difusco/{path_split}/gt{sever_num}s{user_num}u'
    else:
        result_storage_path = f'/root/CPY/DT4SG/difusco/{path_split}/{sever_num}s{user_num}u'
    if not os.path.exists(result_storage_path):
        os.mkdir(result_storage_path)

    if not os.path.exists(result_storage_path + '/reconstructed_grad'):
        grads_reconstruct(original_grads_file_path, result_storage_path + '/reconstructed_grad')
        os.remove(original_grads_file_path)
    angle_calculation(result_storage_path + '/reconstructed_grad', result_storage_path + '/angle')
    avg_angle_calculation(result_storage_path + '/angle', result_storage_path + '/avg_angle')
    avg_angle_for_param(result_storage_path + '/avg_angle', result_storage_path + '/angle_param')
    delete_zero_angle(result_storage_path + '/angle_param')
    show_specific_angles(result_storage_path + '/angle_param')
    param_avg_across_all_epoch(result_storage_path + '/angle_param')

    if not save_information:
        shutil.rmtree(result_storage_path + '/reconstructed_grad')
    shutil.rmtree(result_storage_path + '/angle')
    shutil.rmtree(result_storage_path + '/avg_angle')
