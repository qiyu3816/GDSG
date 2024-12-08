import functools

import torch
from torch import nn
import torch.utils.checkpoint as activation_checkpoint
from torch_sparse import SparseTensor

from models.gnn_encoder import GNNLayer, PositionEmbeddingSine, ScalarEmbeddingSine, ScalarEmbeddingSine1D
from models.nn import (
    zero_module,
    normalization,
)


def run_sparse_layer(layer, out_layer, adj_matrix, edge_index):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward


class GNNDi(nn.Module):
  """
  Discriminative GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               *args, **kwargs):
    super(GNNDi, self).__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    self.node_type_embed = nn.Embedding(2, hidden_dim)
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    edge_dim = 6
    self.edge_attr_embed = nn.Linear(edge_dim, hidden_dim)  # edge attrs in MSCO are not suitable for discrete embedding
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    if not node_feature_only:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)

    if out_channels == 3:
      self.hybrid_task = True
      self.classification_head = nn.Sequential(
        normalization(hidden_dim),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim, 2, kernel_size=1),
      )

      self.regression_head = nn.Sequential(
        normalization(hidden_dim),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim, 1, kernel_size=1),
      )
    else:
      self.hybrid_task = False
      self.out = nn.Sequential(
          normalization(hidden_dim),
          nn.ReLU(),
          # zero_module(
              nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
          # ),
      )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self, x, adj_matrix):
    """
    Args:
        x: Input node coordinates (B x V)
        adj_matrix: Adjacent matrix and edge attributes (B x 6 x V x V), edge_index 1 dim and 5 attr dims
    Returns:
        Updated edge features described with probabilities (B x out_channel x V x V)
    """
    # Embed node & edge features
    x = self.node_embed(self.node_type_embed(x.long()))
    graph = adj_matrix.permute((0, 2, 3, 1))  # (B x V x V x attr_dim)
    e = self.edge_embed(self.edge_attr_embed(graph))
    graph = graph[..., -6].long()

    for layer, out_layer in zip(self.layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      x, e = layer(x, e, graph, edge_matrix=adj_matrix[:, 0, :, :], mode="direct")
      x = x_in + x
      e = e_in + out_layer(e)

    if self.hybrid_task:
      e = torch.cat((self.classification_head(e.permute((0, 3, 1, 2))),
                     self.regression_head(e.permute((0, 3, 1, 2)))), dim=1)
    else:
      e = self.out(e.permute((0, 3, 1, 2)))
    return e

  def sparse_forward(self, x, edge_index):
    """
    Args:
        x: Input node coordinates (V)
        edge_index: Adjacency matrix for the graph ([2 + attr_dim] x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    src_node_num = torch.sum(1 - x).long()
    x = self.node_embed(self.node_type_embed(x.long()))
    graph = edge_index.T[:, 2:]
    e = self.edge_embed(self.edge_attr_embed(graph))
    all_edge_index = edge_index[:2].long()
    real_edge_index = edge_index[2].long()

    x, e = self.sparse_encoding(x, e, all_edge_index, real_edge_index, edge_index[2:])
    e = e.reshape((1, src_node_num, -1, e.shape[-1])).permute((0, 3, 1, 2))
    if self.hybrid_task:
      e = torch.cat((self.classification_head(e), self.regression_head(e)),
                    dim=1).reshape(-1, all_edge_index.shape[1]).permute((1, 0))
    else:
      e = self.out(e).reshape(-1, all_edge_index.shape[1]).permute((1, 0))
    return e

  def sparse_forward_node_feature_only(self, x, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, real_edge_index, edge_attr):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_attr.T,
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, out_layer in zip(self.layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:

        run_sparse_layer_fn = functools.partial(
            run_sparse_layer,
        )

        out = activation_checkpoint.checkpoint(
            run_sparse_layer_fn(layer, out_layer, adj_matrix, edge_index),
            x_in, e_in
        )
        x = out[0]
        e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, real_edge_index=real_edge_index, sparse=True)
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e

  def forward(self, x, edge_index=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, edge_index)
      else:
        raise NotImplementedError
    else:
      if self.sparse:
        return self.sparse_forward(x, edge_index)
      else:
        return self.dense_forward(x, edge_index)


  def freeze_one_layer(self, target_layer):
    for p in target_layer.parameters():
      p.requires_grad = False


  def custom_freeze(self):
    self.freeze_one_layer(self.node_type_embed)
    self.freeze_one_layer(self.node_embed)
    self.freeze_one_layer(self.edge_attr_embed)
    self.freeze_one_layer(self.edge_embed)
    for l1, l2 in zip(self.layers, self.per_layer_out):
      self.freeze_one_layer(l1)
      self.freeze_one_layer(l2)
