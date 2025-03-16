import pathlib
from typing import Dict, Iterable, Literal, Protocol
from fairseq.data import FairseqDataset
import torch
from dpu_utils.utils import RichPath
from tqdm.auto import tqdm

from graph_coder.data.collator import collator
from .registry import register_dataset


class TypilusDatasetConfig(Protocol):
    dataset_root: str
    num_classes: int
    max_tokens: int
    processed_dir: str


@register_dataset("typilus")
class TypilusDataset(FairseqDataset):
    def __init__(self, cfg: TypilusDatasetConfig, split: Literal["train", "valid", "test"] = "train", *args, **kwargs):
        self.cfg = cfg
        self.split = split
        self._data = list(self._load_data())

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collater(self, samples):
        return collator(samples)

    def num_tokens(self, index):
        return self._data[index]["y"].size(-1)
    
    def size(self, index):
        return self._data[index]["y"].size(-1)
    
    def num_tokens_vec(self, indices):
        samples = [self._data[i] for i in indices]
        data = self.collater(samples)
        return data["y"].size(-1)
    
    def _load_data(self) -> Iterable[Dict[str, torch.Tensor]]:
        resolved_path = pathlib.Path(self.cfg.dataset_root).expanduser()
        processed_dir = resolved_path / self.cfg.processed_dir
        split_dir = processed_dir / self.split
        data_path = RichPath.create(str(split_dir))

        for file_path in data_path.iterate_filtered_files_in_dir("data_chunk_*.pkl.gz"):
            yield from tqdm(file_path.read_by_file_suffix(), desc=f"Loading {file_path}")