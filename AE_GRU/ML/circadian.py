import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
import os
from einops import rearrange
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler


def downsample(arr, k=5, accumulate=True):
    new_rows = arr.shape[0] // k
    if accumulate:
        downsampled_arr = np.mean(arr[:new_rows*k].reshape(new_rows, k, -1), axis=1)
    else:
        downsampled_arr = arr[:new_rows*k].reshape(new_rows, k, -1)[:, -1, :]
    return downsampled_arr


from sklearn.preprocessing import StandardScaler, RobustScaler

class UNMDataPreprocessing(object):
    def __init__(
        self,
        file_path=None,
        down_sample_rate=1,
        num_label=24,
        encode_label=True,
    ):
        self.down_sample_rate = down_sample_rate
        self.num_label = num_label
        
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self.train_outputs, self.train_inputs = None, None
        self.test_outputs, self.test_inputs = None, None
        self.train_labels, self.test_labels = None, None
        
        if file_path is not None:
            self.read_file(file_path, encode_label)

    def read_file(self, file_path, encode_label=True):
        df = pd.read_csv(file_path)
        dr = self.down_sample_rate

        inputs = np.stack([df['mean_light_3'].values], axis=1)
        outputs = np.stack([df['raw_actigraphy'].values], axis=1)
        labels = np.stack([df['raw_theta'].values], axis=1)
        if encode_label:
            labels = self.encode_label(labels)

        # inputs = self.input_scaler.fit_transform(inputs)
        # outputs = self.output_scaler.fit_transform(outputs)

        start_idx = int(1440*3)
        split_idx = int(1440*12)
        train_inputs, train_outputs = downsample(inputs[start_idx:split_idx], dr), downsample(outputs[start_idx:split_idx], dr)
        test_inputs, test_outputs = downsample(inputs[split_idx:], dr), downsample(outputs[split_idx:], dr)
        train_labels, test_labels = downsample(labels[start_idx:split_idx], dr, False), downsample(labels[split_idx:], dr, False)
        
        assert train_inputs.shape[0] == train_outputs.shape[0] == train_labels.shape[0]
        assert test_inputs.shape[0] == test_outputs.shape[0] == test_labels.shape[0]

        train_inputs = self.input_scaler.fit_transform(train_inputs)
        train_outputs = self.output_scaler.fit_transform(train_outputs)
        test_inputs = self.input_scaler.transform(test_inputs)
        test_outputs = self.output_scaler.transform(test_outputs)

        self.train_outputs = torch.from_numpy(train_outputs).unsqueeze(0).permute(0, 2, 1).float()
        self.test_outputs = torch.from_numpy(test_outputs).unsqueeze(0).permute(0, 2, 1).float()
        self.train_inputs = torch.from_numpy(train_inputs).unsqueeze(0).permute(0, 2, 1).float()
        self.test_inputs = torch.from_numpy(test_inputs).unsqueeze(0).permute(0, 2, 1).float()
        self.train_labels = torch.from_numpy(train_labels).T #.long()
        self.test_labels = torch.from_numpy(test_labels).T #.long()

        return self.train_inputs, self.train_outputs, self.train_labels, self.test_inputs, self.test_outputs, self.test_labels

    def get_data(self):
        return self.train_inputs, self.train_outputs, self.train_labels, self.test_inputs, self.test_outputs, self.test_labels

    def encode_label(self, label):
        thresholds = np.linspace(np.min(label), np.max(label), self.num_label + 1)
        new_labels = np.zeros_like(label)

        for i in range(self.num_label):
            new_labels[(label >= thresholds[i]) & (label < thresholds[i+1])] = i

        return new_labels


class CircadianDataPreprocessing(object):
    def __init__(
        self,
        num_split = 5,
        obs_dim = 10,
        mat_c = None,
        mat_d = None
    ):
        self.num_split = num_split
        self.obs_dim = obs_dim
        self.mat_c = torch.randn(2, obs_dim) if mat_c is None else mat_c
        self.mat_c[-1, :] = 0.
        self.mat_d = torch.randn(1, obs_dim) if mat_d is None else mat_d
        self.label_encoder = LabelEncoder()
        self.thresholds = None
        
    def transform(
        self,
        data
    ):
        x = torch.tensor(data['x']).float()
        xc = torch.tensor(data['xc']).float()
        u = torch.tensor(data['u']).float()
        t = self.get_state_label(x, xc, fit=False)
        
        return self._transform(x, xc, u, t)
    
    def fit_transform(
        self,
        data
    ):
        x = torch.tensor(data['x']).float()
        xc = torch.tensor(data['xc']).float()
        u = torch.tensor(data['u']).float()
        t = self.get_state_label(x, xc, fit=True)
        
        return self._transform(x, xc, u, t)
        
    def _transform(self, x, xc, u, t):
        state = torch.stack([x, xc], dim=-1)
        u = u.unsqueeze(-1)
        # t = t.unsqueeze(-1)
        
        obs, u, state = self.create_linear_observation(state, u)
        # t = rearrange(t, 'n t c -> n c t')
        return obs, u, state, t
        
        
    def create_linear_observation(self, state, u):
        obs = state @ self.mat_c + u @ self.mat_d
        
        obs = rearrange(obs, 'n t c -> n c t')
        u = rearrange(u, 'n t c -> n c t')
        state = rearrange(state, 'n t c -> n c t')
    
        return obs, u, state
        
    def get_state_label(
        self,
        x,
        xc,
        fit=False
    ):
        x_label = self.label_array(x, self.num_split, fit=fit)
        xc_label = self.label_array(xc, self.num_split, fit=fit)
        t_label = np.zeros_like(x_label)
        num_split = self.num_split
        
        for i in range(t_label.shape[0]):
            t_label[i] = x_label[i] * num_split + xc_label[i]
        
        if fit:
            t_label = self.label_encoder.fit_transform(t_label)
        else:
            t_label = self.label_encoder.transform(t_label)
        
        t_label = rearrange(t_label, '(n t) -> n t', n=x.shape[0])
        return torch.tensor(t_label).long()

    
    def label_array(self, arr, num_split=5, fit=False):
        """
        Divide the array into 5 groups based on value and label samples in the array with the group index.

        Parameters:
        arr (list or numpy array): The input array.

        Returns:
        list: A list of labels corresponding to the group index for each element in the array.
        """
        
        # Convert to numpy array for easier manipulation
        arr = rearrange(np.array(arr), 'n t -> (n t)')

        # Calculate number of elements per group
        num_elements_per_group = len(arr) // num_split

        # Use numpy's percentile function to find the thresholds for the groups
        if fit:
            self.thresholds = [np.percentile(arr, i * 100/num_split) for i in range(0, num_split + 1)]

        # Initialize an array to store the labels
        labels = np.zeros(len(arr), dtype=int)

        # Assign labels based on the thresholds
        for i in range(num_split):
            labels[(arr >= self.thresholds[i]) & (arr < self.thresholds[i + 1])] = i

        return labels

    
# class CircadianDataset(Dataset):
#     def __init__(self, 
#                  observations,
#                  actions,
#                  timestamp,
#                  labels=None,
#                  horizon=10,
#                  noise=False,
#                  num_time_label=10
#                 ):
#         '''
#         Observation dim: [n_sample, n_channel, n_timestamp]
#         Action dim: [n_sample, n_channel, n_timestamp]
#         '''
        
#         self.horizon = horizon
#         self.target = rearrange(copy.deepcopy(observations[:, :, horizon:]), 'n c t -> (n t) c') if labels is None else rearrange(labels, 'n c t -> (n t) c')

#         if noise:
#             observations += torch.randn_like(observations) * 0.1 * observations.std(dim=-1, keepdim=True)
#         self.obs_act_traj = torch.cat([observations, actions], dim=1)
#         obs_window = rearrange(self.obs_act_traj.unfold(-1, horizon, 1), 'n c t h -> n t c h')
        
#         self.obs = rearrange(obs_window[:, :-1, :, :], 'n t c h -> (n t) (c h)')
#         self.next_obs = rearrange(obs_window[:, 1:, :, :], 'n t c h -> (n t) (c h)')
        
#         self.action = rearrange(actions[:, :, horizon-1:-1], 'n c t -> (n t) c')
#         self.next_action = rearrange(actions[:, :, horizon:], 'n c t -> (n t) c')
        
#         # n_time = np.ceil(timestamp.max()/num_time_label)
#         # timestamp = timestamp // n_time
#         self.t = rearrange(timestamp[:, :, horizon-1:-1], 'n c t -> (n t) c')
#         self.next_t = rearrange(timestamp[:, :, horizon:], 'n c t -> (n t) c')
        
#         assert self.obs.shape[0] == self.action.shape[0] == self.target.shape[0]
        
#     def __len__(self):
#         return self.obs.shape[0]
    
#     def __getitem__(self, idx):
#         transition = (self.obs[idx], self.action[idx], self.next_obs[idx], self.next_action[idx], self.t[idx])
#         return transition, self.target[idx]
    
    
class CircadianDataset(Dataset):
    def __init__(self, 
                 observations,
                 actions,
                 timestamp,
                 labels=None,
                 horizon=10,
                 noise=False,
                 num_time_label=10,
                 step=1
                ):
        '''
        Observation dim: [n_sample, n_channel, n_timestamp]
        Action dim: [n_sample, n_channel, n_timestamp]
        '''
        
        self.horizon = horizon
        
        if noise:
            observations += torch.randn_like(observations) * 0.1 * observations.std(dim=-1, keepdim=True)
        # obs_act_traj = torch.cat([observations, actions], dim=1)
        # obs_window = rearrange(obs_act_traj.unfold(-1, horizon, 1), 'n c t h -> n t c h')
        self.obs = observations.unfold(-1, horizon, step)
        self.obs = rearrange(self.obs, 'n c t h -> (n t) h c')
        self.action = actions.unfold(-1, horizon, step)
        self.action = rearrange(self.action, 'n c t h -> (n t) h c')
        self.t = timestamp.unfold(-1, horizon, step)
        self.t = rearrange(self.t, 'n t h -> (n t) h')
        
        assert self.obs.shape[0] == self.action.shape[0] == self.t.shape[0]
        
    def __len__(self):
        return self.obs.shape[0]
    
    def __getitem__(self, idx):
        transition = (self.obs[idx], self.action[idx], self.t[idx])
        return transition