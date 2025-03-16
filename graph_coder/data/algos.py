import torch
import torch.utils.data


def eig(sym_mat):
    # (sorted) eigenvectors with torch
    eigval, eigvec = torch.linalg.eigh(sym_mat)
    
    # for eigval, take abs because sometimes the first eigenvalue approaches 0 from the negative
    eigval, indices = torch.sort(torch.abs(torch.real(eigval)))
    eigvec = eigvec[:, indices]
    
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def lap_eig(dense_adj: torch.BoolTensor, number_of_nodes: int, in_degree: torch.Tensor):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    dense_adj = dense_adj.detach().float()
    in_degree = in_degree.detach().float()
    # Laplacian
    A = dense_adj
    N = torch.diag(torch.clamp(in_degree, min=1) ** -0.5)
    L = torch.eye(number_of_nodes, device=dense_adj.device) - N @ A @ N

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]
