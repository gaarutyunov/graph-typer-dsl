"""
Modified from https://github.com/microsoft/Graphormer
"""
import math

import torch

from .algos import lap_eig


def convert_to_single_emb(x: torch.Tensor, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item: dict[str, torch.Tensor], mask_ratio: float = 0.5, mask: bool = True):
    edge_int_feature, edge_index, node_int_feature = item["edge_attr"], item["edge_index"], item["x"]
    if len(edge_int_feature.size()) == 1:
        edge_int_feature = edge_int_feature[:, None]
    if len(node_int_feature.size()) == 1:
        node_int_feature = node_int_feature[:, None]

    node_data = convert_to_single_emb(node_int_feature)
    edge_data = convert_to_single_emb(edge_int_feature)

    N = node_int_feature.size(0)
    dense_adj = torch.zeros([N, N], dtype=torch.bool)
    dense_adj[edge_index[0, :], edge_index[1, :]] = True
    in_degree = dense_adj.long().sum(dim=1).view(-1)
    lap_eigvec, lap_eigval = lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]
    lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)

    if mask:
        nodes_with_labels: torch.BoolTensor = item["y"] != -100
        indices = nodes_with_labels.nonzero().squeeze()
        num_to_mask = math.ceil(nodes_with_labels.sum().item() * mask_ratio)
        perm = torch.randperm(indices.size(0))
        node_mask = indices[perm[:num_to_mask]]
        token_count = node_data.size(0) + edge_data.size(0)
        token_mask = torch.zeros(token_count, dtype=torch.bool)
        token_mask[node_mask] = True
        item["masked_tokens"] = token_mask

    item["node_data"] = node_data
    item["edge_data"] = edge_data
    item["edge_index"] = edge_index
    item["in_degree"] = in_degree
    item["out_degree"] = in_degree  # for undirected graph
    item["lap_eigvec"] = lap_eigvec
    item["lap_eigval"] = lap_eigval

    return item
