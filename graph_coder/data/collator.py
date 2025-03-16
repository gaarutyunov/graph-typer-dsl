import torch
import torch.nn.functional as F


@torch.no_grad()
def collator(
        items,
        multi_hop_max_dist=20,
        spatial_pos_max=20,
):
    (
        edge_index,
        edge_data,
        node_data,
        in_degree,
        out_degree,
        lap_eigvec,
        lap_eigval,
        ys,
    ) = zip(*[
        (
            item['edge_index'],
            item['edge_data'],
            item['node_data'],
            item['in_degree'],
            item['out_degree'],
            item['lap_eigvec'],
            item['lap_eigval'],
            item['y'],
        )
        for item in items
    ])

    node_num = torch.tensor([i.size(0) for i in node_data])
    edge_num = torch.tensor([i.size(0) for i in edge_data])
    max_n = max(node_num)
    seq_len = max(node_num + edge_num)

    y = torch.cat([F.pad(i[None, ...], (0, seq_len - i.size(0)), value=-100) for i in ys])
    edge_index = torch.cat(edge_index, dim=1)  # [2, sum(edge_num)]
    edge_data = torch.cat(edge_data) + 1  # [sum(edge_num), De], +1 for nn.Embedding with pad_index=0
    node_data = torch.cat(node_data) + 1  # [sum(node_num), Dn], +1 for nn.Embedding with pad_index=0
    in_degree = torch.cat(in_degree) + 1  # [sum(node_num),], +1 for nn.Embedding with pad_index=0
    out_degree = torch.cat(out_degree) + 1  # [sum(node_num),], +1 for nn.Embedding with pad_index=0

    # [sum(node_num), Dl] = [sum(node_num), max_n]
    lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec])
    lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval])

    return dict(
        edge_index=edge_index,
        edge_data=edge_data,
        node_data=node_data,
        in_degree=in_degree,
        out_degree=out_degree,
        lap_eigvec=lap_eigvec,
        lap_eigval=lap_eigval,
        y=y,
        node_num=node_num[:, None],
        edge_num=edge_num[:, None],
    )
