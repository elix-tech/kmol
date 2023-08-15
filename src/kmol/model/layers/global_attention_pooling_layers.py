import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(GlobalAttentionPooling, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x shape: batch_size x sequence_length x input_dim
        attention_scores = self.attention_network(x)
        # attention_scores shape: batch_size x sequence_length x 1

        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights shape: batch_size x sequence_length x 1

        weighted_sum = torch.sum(x * attention_weights, dim=1)
        # weighted_sum shape: batch_size x input_dim

        return weighted_sum


class GlobalAttentionPooling2D(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(GlobalAttentionPooling2D, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Conv2d(input_dim, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(attention_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: batch_size x input_dim x N x N
        attention_scores = self.attention_network(x)
        # attention_scores shape: batch_size x 1 x N x N

        attention_weights = F.softmax(attention_scores.flatten(2), dim=-1).view_as(attention_scores)
        # attention_weights shape: batch_size x 1 x N x N

        weighted_sum = torch.sum(x * attention_weights, dim=(2,3))
        # weighted_sum shape: batch_size x input_dim

        return weighted_sum


