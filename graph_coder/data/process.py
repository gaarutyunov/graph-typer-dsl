
import logging
import torch
from tqdm.auto import tqdm

from graph_coder.data.wrapper import preprocess_item
from dpu_utils.utils import RichPath


_necessary_keys = ['cg_node_label_token_ids', 'cg_edges', 'target_node_idxs', 'variable_target_class']

def process_file(file_path: RichPath, output_file_path: RichPath, mask: bool, max_tokens: int, position: int = 0) -> None:
    result = []

    for data_chunk in tqdm(file_path.read_by_file_suffix(), desc=f"Loading {file_path}", position=position):
        # check for necessary keys, skip if empty
        if not all(k in data_chunk for k in _necessary_keys):
            continue

        n_edges = 0

        for edges in data_chunk['cg_edges']:
            n_edges += edges.shape[0]

        num_tokens = data_chunk['cg_node_label_token_ids'].shape[0] + n_edges

        if num_tokens > max_tokens:
            continue

        edge_index = torch.zeros((2, n_edges), dtype=torch.long)
        edge_attr = torch.zeros((n_edges, 1), dtype=torch.long)

        start_offset = 0

        for i, edges in enumerate(data_chunk['cg_edges']):
            if edges.shape[0] > 0:  # Make sure there are edges of this type
                edge_index[:, start_offset:start_offset + edges.shape[0]] = torch.tensor(edges, dtype=torch.long).t()
                edge_attr[start_offset:start_offset + edges.shape[0], 0] = i
                start_offset += edges.shape[0]

        # Create node labels tensor
        y = torch.full((data_chunk['cg_node_label_token_ids'].shape[0],), -100)
        idx = torch.tensor(data_chunk['target_node_idxs'], dtype=torch.long)
        y[idx] = torch.tensor(data_chunk['variable_target_class'], dtype=torch.long)

        # Create dict
        data = dict(
            x=torch.tensor(data_chunk['cg_node_label_token_ids'], dtype=torch.long),
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        data = preprocess_item(data, mask=mask)

        result.append(data)

    logging.info(f"Writing {len(result)} items to {output_file_path}")

    output_file_path.save_as_compressed_file(result)
