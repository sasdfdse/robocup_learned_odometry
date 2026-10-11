# Copyright 2024 Booster Robotics Technology Co., Ltd. ("Booster Robotics")
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
# Taken from humanoid-soccer model/odom.py.
# Modified for robocup_learned_odometry: removed the unused tanh_encoder_output argument
# and debug prints, added odom_input_dim(). The network architecture is unchanged.
import torch.nn as nn


class StateHistoryEncoder(nn.Module):

    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super().__init__()
        self.activation_fn = activation_fn
        self.input_size = input_size
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4), self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1), self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2), self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2), self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2), self.activation_fn,
                nn.Flatten())
        else:
            raise ValueError("tsteps must be 10, 20 or 50")

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class OdomModel(nn.Module):

    def __init__(self, num_prop, hist_len):
        super().__init__()
        self.num_latent = 64
        self.hist_encoder = StateHistoryEncoder(nn.ELU(), num_prop, hist_len, self.num_latent)
        self.trunk = nn.Sequential(
            nn.Linear(self.num_latent, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 4),
        )

    def forward(self, obs):
        latent_feat = self.hist_encoder(obs)
        out = self.trunk(latent_feat)
        return out


def odom_input_dim(num_actions):
    return 6 + 3 * num_actions
