import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
import os
import glob
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from pytorch_metric_learning.losses import NTXentLoss, ContrastiveLoss
from pytorch_metric_learning.distances import LpDistance
from torch.utils.tensorboard import SummaryWriter
import json

from circadian import CircadianDataPreprocessing, CircadianDataset, UNMDataPreprocessing, downsample
from utils import plot_obs_prediction, plot_state_trajectory, create_image_grid

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda')


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[64, 64],
        activation=nn.ReLU()
    ):
        super().__init__()
        
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(activation)
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        return self.layers(x)


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        det_state_dim=2,
        stoch_state_dim=2,
        encoder_hidden_dim=64,
        window_len=10,
        activation=nn.ReLU()
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.det_state_dim = det_state_dim
        self.window_len = window_len
        self.encoder_hidden_dim = encoder_hidden_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            MLP((obs_dim + action_dim) * self.window_len, stoch_state_dim, activation=activation)
        )
        self.prior_mlp = MLP(det_state_dim+stoch_state_dim, det_state_dim, hidden_dims=[32, 32], activation=activation)

        self.rnn = nn.GRUCell(action_dim+stoch_state_dim, det_state_dim)
        self.post_mlp = MLP(det_state_dim, stoch_state_dim, hidden_dims=[16, 16], activation=activation)

        self.decoder = MLP(stoch_state_dim+action_dim, obs_dim)

        distance = LpDistance(normalize_embeddings=False, p=2, power=1)
        self.contrastive_func = ContrastiveLoss(pos_margin=0.1, neg_margin=0.2, distance=distance)

    def forward(self, batch, mask=None):
        obs, action, t_label = batch
        batch_size = obs.shape[0]
        
        obs_action_traj = torch.cat([obs, action], dim=-1)
        target_obs_traj = obs[:, self.window_len-1:, :]
        action_traj = action[:, self.window_len-1:, :]
        labels = t_label[:, self.window_len-1:]
        x0 = torch.zeros(batch_size, self.det_state_dim, device=obs.device)

        priors = self.encode(obs_action_traj)
        priors = rearrange(priors, '(b n) d -> b n d', b=batch_size)
        s0 = priors[:, 0, :] #.contiguous()

        x_traj = [x0]
        posts = [s0]
        # post_s_dists = [prior_logits[:, 0, :]]
        mask = torch.randint(high=2, size=(action_traj.shape[0], action_traj.shape[1]), device=obs.device) if mask is None else mask[:, :action_traj.shape[1]]
        
        for t in range(action_traj.shape[1]-1):
            # Compute next determinstic state ht
            mt = mask[:, [t]]
            st = mt * priors[:, t, :] + (1 - mt) * posts[-1]
            at = action_traj[:, t, :]
            xt = x_traj[-1]
            # ht = self.prior_mlp(torch.cat([xt, st], dim=-1))
            xt = self.rnn(torch.cat([at, st], dim=-1), xt)

            st = self.post_mlp(xt)
            x_traj.append(xt)
            posts.append(st)
            
        x_traj = torch.stack(x_traj, dim=1)
        posts = torch.stack(posts, dim=1)

        # Compute observations
        post_obs_pred = self.decode(torch.cat([posts, action_traj], dim=-1))
        # prior_obs_pred = self.decode(torch.cat([priors, action_traj], dim=-1))
        return {
            # 'prior_obs_pred': prior_obs_pred,
            'post_obs_pred': post_obs_pred,
            'priors': priors,
            'posts': posts,
            'x_traj': x_traj,
            'obs_target': target_obs_traj,
            't_label': labels
        }

    def loss(self, ret, contrastive=True):
        dyn_loss = nn.functional.mse_loss(ret['posts'], ret['priors'].detach())
        rep_loss = nn.functional.mse_loss(ret['posts'].detach(), ret['priors'])
        rnn_reconstruct = torch.nn.functional.mse_loss(ret['post_obs_pred'][:, 1:], ret['obs_target'][:, 1:])
        # encoder_reconstruct = torch.nn.functional.mse_loss(ret['prior_obs_pred'], ret['obs_target'])
        if contrastive:
            state_contrastive = self.state_contrastive(ret['x_traj'], ret['t_label'])
        else:
            state_contrastive = 0.

        return {
            'dyn_loss': dyn_loss,
            'rep_loss': rep_loss,
            'obs_reconstruct': rnn_reconstruct, #+ encoder_reconstruct,
            'state_contrastive': state_contrastive
        }

    def state_contrastive(self, state, label):
        loss = sum([self.contrastive_func(xn, ln) for xn, ln in zip(state, label.long())])
        return loss / state.shape[0]
        
        # ref = torch.stack([torch.sin(label.float()), torch.cos(label.float())], dim=-1)
        # return nn.functional.mse_loss(state, ref)

    def encode(self, x):
        x = x.unfold(1, self.window_len, 1) #.flatten(start_dim=2)
        x = rearrange(x, 'b n c t -> (b n) c t')
        x = self.encoder(x)
        return x

    def decode(self, x):
        return self.decoder(x)
    

def get_args(message=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=10, choices=[3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--window_size', type=int, default=512)
    parser.add_argument('--down_sample_rate', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exp_num', type=int, default=0)

    if message is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(message)
    return args


def get_data(args, encode_label=True):
   

    data_preprocessor = UNMDataPreprocessing(
        f'data/unm/unm_data_{args.subject_id}.csv',
        down_sample_rate=args.down_sample_rate,
        encode_label=encode_label
    )
    train_input, train_obs, train_label, test_input, test_obs, test_label = data_preprocessor.get_data()

    train_dataset = CircadianDataset(train_obs, train_input, train_label, horizon=args.window_size, noise=False)
    # test_dataset = CircadianDataset(test_obs, test_input, test_label, horizon=args.window_size, noise=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return (train_input, train_obs, train_label, test_input, test_obs, test_label), (train_loader, None)


def create_mask(k, length, batch_size, device):
    # Create a basic pattern [1, 0, 0, ..., 0] with length k
    pattern = torch.zeros(k, device=device)
    pattern[0] = 1

    # Repeat the pattern to fill the tensor
    return pattern.repeat(batch_size, int(length/k)+1)


def k_step_prediction(model, test_data, save_dir='./', start_ts=0):
    test_obs, test_input, test_label = test_data

    test_dataset = CircadianDataset(test_obs, test_input, test_label, horizon=test_obs.shape[-1], noise=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # text_x_gt = test_state
    result_dict = {
        'predict_horizon': [],
        'mse': []
    }

    trajectory_dict = dict()

    for time_horizon in [5, 15, 60, 120, 180, 240, 300, 360, 420, 480, 24*60, 48*60]:
        k = int(time_horizon / 5)

        test_y_preds = []
        test_y_targets = []
        test_y_error = []
        test_x_preds = []
        test_u = []
        test_labels = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(test_loader)):
                # print(batch_id, batch[0].shape, batch[1].shape, batch[2].shape)
                batch = [x.to(device) for x in batch]
                obs_traj, u_traj, ts_traj = batch
                mask = create_mask(k, obs_traj.shape[1], obs_traj.shape[0], device)
                # print(mask.shape)
                output = model(batch, mask=mask)
                loss_dict = model.loss(output)
        
                test_y_error.append(loss_dict['obs_reconstruct'].cpu().numpy())
                test_y_preds.append(output['post_obs_pred'].squeeze(0).cpu().numpy().T)
                test_y_targets.append(output['obs_target'].squeeze(0).cpu().numpy().T)
                test_x_preds.append(output['x_traj'].squeeze(0).cpu().numpy().T)
                test_u.append(u_traj.squeeze(0).cpu().numpy())
                test_labels.append(ts_traj.squeeze(0).cpu().numpy())
        
            for y_error, yp, yt in zip(test_y_error, test_y_preds, test_y_targets):
                y_error = np.mean(np.power(yp[:, start_ts:] - yt[:, start_ts:], 2))
                print(f"Prediction Error: {y_error:.4f}")
                if time_horizon < 60:
                    title = f"Horizon: {time_horizon} minutes, MSE: {y_error:.4f} (a.u.)"
                else:
                    title = f"Horizon: {int(time_horizon/60)} hour(s), MSE: {y_error:.4f} (a.u.)"
                obs_fig = plot_obs_prediction(
                    yp[:, start_ts:], 
                    yt[:, start_ts:], 
                    show=False,
                    title=title
                )
                # obs_fig.suptitle(f"Horizon: {time_horizon}, MSE: {y_error:.4f}", fontsize=16)
                obs_fig.savefig(f"{save_dir}/test_obs-{time_horizon}.png", bbox_inches='tight', dpi=300)
                plt.close(obs_fig)

                if 'y_target' not in trajectory_dict.keys():
                    trajectory_dict['ts'] = np.arange(yt.shape[-1])/12
                    trajectory_dict['y_target'] = yt.flatten()
                    trajectory_dict['u'] = np.array(test_u).flatten()[-yt.flatten().shape[0]:]
                trajectory_dict[f'y_pred_{time_horizon}'] = yp.flatten()
            
            x_preds = np.concatenate(test_x_preds, axis=1)[:, start_ts:]
            x1_range = np.max(x_preds[0]) - np.min(x_preds[0])
            x2_range = np.max(x_preds[1]) - np.min(x_preds[1])
            xlims = [np.min(x_preds[0])-0.05*x1_range, np.max(x_preds[0])+0.05*x1_range]
            ylims = [np.min(x_preds[1])-0.05*x2_range, np.max(x_preds[1])+0.05*x2_range]
            
            for i, (xt, ut, t_label) in enumerate(tqdm(zip(test_x_preds, test_u, test_labels))):
                
                if time_horizon < 60:
                    title = f"Horizon: {time_horizon} minutes"
                else:
                    title = f"Horizon: {int(time_horizon/60)} hour(s)"
                fig = plot_state_trajectory(
                    xt[:, start_ts:], 
                    ut[start_ts:], 
                    show=False, 
                    xlim=xlims, 
                    ylim=ylims,
                    title=title
                )
                # fig.suptitle(f"Horizon: {time_horizon}", fontsize=16)
                fig.savefig(f"{save_dir}/test_state-{time_horizon}.png", bbox_inches='tight', dpi=300)
                plt.close(fig)

                trajectory_dict[f'x1_{time_horizon}'] = xt[0, :]
                trajectory_dict[f'x2_{time_horizon}'] = xt[1, :]
        
            result_dict['predict_horizon'].append(time_horizon)
            result_dict['mse'].append(y_error)

    for key, val in trajectory_dict.items():
        print(key, val.shape)

    return pd.DataFrame(result_dict), pd.DataFrame(trajectory_dict)


def main(args):
    dataset = 'unm'
    log_dir = f'runs/rssm-unm-final/subject-{args.subject_id}'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    exp_num = len(glob.glob(f"{log_dir}/exp*"))

    try:
        os.makedirs(f'{log_dir}/exp{exp_num}')
    except:
        pass
    writer = SummaryWriter(f'{log_dir}/exp{exp_num}/')

    # Load data
    raw_data, loaders = get_data(args)
    train_input, train_obs, train_label, test_input, test_obs, test_label = raw_data
    train_loader, test_loader = loaders

    print(train_input.shape, train_obs.shape, train_label.shape)

    _, state_dim, episode_len = train_obs.shape
    _, action_dim, _ = train_input.shape

    # Create Model
    lr = 1e-3
    n_epoch = 200
    det_state_dim = 2
    stoch_state_dim = 16
    encoder_hidden_dim = 64
    window_len = 10

    model = RSSM(
        obs_dim=train_obs.shape[1],
        action_dim=train_input.shape[1],
        det_state_dim=det_state_dim,
        stoch_state_dim=stoch_state_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        window_len=window_len,
        activation=nn.ReLU()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)

    model.train()

    loss_weights = {
        'dyn_loss': 0.2,
        'rep_loss': 1.5,
        'obs_reconstruct': 1.,
        'state_contrastive': 0.
    }

    print(f"Start Training: Subject {args.subject_id}")
    print(f'{log_dir}/exp{exp_num}')
    print(device)

    # Start Training
    for epoch in tqdm(range(n_epoch), desc='epoch'):
        epoch_loss = dict()
        
        # for batch_id, batch in enumerate(tqdm(train_loader, leave=False, desc='batch')):
        for batch_id, batch in enumerate(train_loader):
            batch = [x.to(device) for x in batch]
            obs_traj, u_traj, ts_traj = batch
            k = np.random.randint(3, obs_traj.shape[1]+1)
            mask = create_mask(k, obs_traj.shape[1], obs_traj.shape[0], device)
            output = model(batch, mask=mask)
            loss_dict = model.loss(output, contrastive=loss_weights['state_contrastive'] > 0)
            loss = 0.

            loss_weights['rep_loss'] = epoch / 10
            for key, val in loss_dict.items():
                if val <= 0.:
                    continue
                if key in epoch_loss:
                    epoch_loss[key] += val
                else:
                    epoch_loss[key] = val
                loss += val * loss_weights[key]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_loss = {key: epoch_loss[key]/len(train_loader) for key in epoch_loss}
        
        total_loss = 0.
        for key, val in epoch_loss.items():
            writer.add_scalar(f"Loss/{key}", val, epoch)
            total_loss += val * loss_weights[key]
        writer.add_scalar("Loss/total", total_loss, epoch)
        
    writer.close()

    # Write configurations to JSON file
    config_dict = {
        'model': {
            'model_path':  f"{log_dir}/exp{exp_num}/model.pt",
            'state_dict_path': f"{log_dir}/exp{exp_num}/state_dict.pt"
        },
        'loss': loss_weights,
        'config': {
            'window_size': args.window_size,
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'n_epoch': n_epoch,
            'det_state_dim': det_state_dim,
            'stoch_state_dim': stoch_state_dim,
            'encoder_hidden_dim': encoder_hidden_dim,
            'window_len': window_len
        }
    }

    with open(f'{log_dir}/exp{exp_num}/config.json', 'w') as file:
        json.dump(config_dict, file, indent=4)

    torch.save(model, f"{log_dir}/exp{exp_num}/model.pt")
    torch.save(model.state_dict(), f"{log_dir}/exp{exp_num}/state_dict.pt")


    # =============================================
    # Multi-step Prediction Evaluation
    eval_obs = torch.cat([train_obs, test_obs], dim=-1)
    eval_input = torch.cat([train_input, test_input], dim=-1)
    eval_label = torch.cat([train_label, test_label], dim=-1)
    eval_data = (eval_obs, eval_input, eval_label)
    eval_df, eval_traj_df = k_step_prediction(model, eval_data, save_dir=f"{log_dir}/exp{exp_num}", start_ts=int(9*1440/5))
    eval_df.to_csv(f"{log_dir}/exp{exp_num}/eval.csv", index=False)
    eval_traj_df.to_csv(f"{log_dir}/exp{exp_num}/eval_traj.csv", index=False)


    #==============================================
    # Create figures
    image_paths = [f"{log_dir}/exp{exp_num}/test_state-{k}.png" for k in [15, 60, 120, 180, 240, 300, 360, 420, 480, 24*60]]

    # Create the image grid
    image_grid = create_image_grid(image_paths)

    # Save the grid image or do further processing
    image_grid.save(f"{log_dir}/exp{exp_num}/state_grid.png")


    image_paths = [f"{log_dir}/exp{exp_num}/test_obs-{k}.png" for k in [15, 60, 120, 180, 240, 300, 360, 420, 480, 24*60]]

    # Create the image grid
    image_grid = create_image_grid(image_paths, grid_size=(5, 2))

    # Save the grid image or do further processing
    image_grid.save(f"{log_dir}/exp{exp_num}/obs_grid.png")


def test(args):
    log_dir = f'runs/rssm-unm-final/subject-{args.subject_id}'
    exp_num = args.exp_num

    raw_data, loaders = get_data(args)
    train_input, train_obs, train_label, test_input, test_obs, test_label = raw_data
    train_loader, test_loader = loaders

    _, state_dim, episode_len = train_obs.shape
    _, action_dim, _ = train_input.shape

    # Create Model
    lr = 1e-3
    n_epoch = 500
    det_state_dim = 2
    stoch_state_dim = 16
    encoder_hidden_dim = 64
    window_len = 10

    model = RSSM(
        obs_dim=train_obs.shape[1],
        action_dim=train_input.shape[1],
        det_state_dim=det_state_dim,
        stoch_state_dim=stoch_state_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        window_len=window_len,
        activation=nn.ReLU()
    ).to(device)
    model.load_state_dict(torch.load(f"{log_dir}/exp{exp_num}/state_dict.pt"))


    # =============================================
    # Multi-step Prediction Evaluation
    test_obs = torch.cat([train_obs, test_obs], dim=-1)
    test_input = torch.cat([train_input, test_input], dim=-1)
    test_label = torch.cat([train_label, test_label], dim=-1)
    test_data = (test_obs, test_input, test_label)
    eval_df, eval_traj_df = k_step_prediction(model, test_data, save_dir=f"{log_dir}/exp{exp_num}", start_ts=int(9*1440/5))
    # eval_df, eval_traj_df = k_step_prediction(model, test_data, save_dir=f"{log_dir}/exp{exp_num}", start_ts=0)
    eval_df.to_csv(f"{log_dir}/exp{exp_num}/eval.csv")
    eval_traj_df.to_csv(f"{log_dir}/exp{exp_num}/eval_traj.csv", index=False)


    # #==============================================
    # # Create figures
    # image_paths = [f"{log_dir}/exp{exp_num}/test_state-{k}.png" for k in [15, 60, 120, 180, 240, 300, 360, 420, 480, 24*60]]

    # # Create the image grid
    # image_grid = create_image_grid(image_paths)

    # # Save the grid image or do further processing
    # image_grid.save(f"{log_dir}/exp{exp_num}/state_grid.png")


    # image_paths = [f"{log_dir}/exp{exp_num}/test_obs-{k}.png" for k in [15, 60, 120, 180, 240, 300, 360, 420, 480, 24*60]]

    # # Create the image grid
    # image_grid = create_image_grid(image_paths, grid_size=(5, 2))

    # # Save the grid image or do further processing
    # image_grid.save(f"{log_dir}/exp{exp_num}/obs_grid.png")

if __name__ == '__main__':
    args = get_args()

    if args.test:
        test(args)
    else:
        main(args)