import logging
import math
import os
import pathlib
from typing import Union, List, Tuple, Optional, Literal, Dict, Sized, Iterable, Any

from fairseq.distributed.utils import get_global_rank, get_global_world_size
import numpy as np
import torch
from dpu_utils.utils import RichPath, ChunkWriter
from fairseq.data import FairseqIterableDataset
from torch_geometric.data import Data
from tqdm.auto import tqdm

from graph_coder.data.utils import sample_to_nx, nx_to_data
from tokengt.data import register_dataset
from .masked_collator import collator
from tokengt.data.wrapper import preprocess_item
from typilus.model.model import read_data_chunks

logger = logging.getLogger(__name__)


def filter_data(data: Data, max_size: int) -> bool:
    return data.y.size(0) + data.edge_index.size(1) > max_size


@register_dataset("pldi2020")
def pldi2020(cfg, split: Literal["train", "test", "valid"] = "train", **kwargs):
    return PLDI2020Dataset(
        os.path.expanduser(cfg.dataset_root),
        split=split,
        num_classes=cfg.num_classes,
        sizes={
            "train": 7,
            "valid": 1,
            "test": 2
        },
        max_tokens=cfg.max_tokens,
        num_workers=cfg.num_data_workers,
        processed_dir=cfg.processed_dir,
        mask_ratio=getattr(cfg, "mask_ratio", 0.5),
        batch_size=getattr(cfg, "batch_size", 4),
        **kwargs
    ).process()


class PLDI2020Dataset(FairseqIterableDataset, Sized):
    def __len__(self):
        sizes_mask = self.sizes <= self._max_tokens

        length = sizes_mask.sum().item() // self._world_size

        return length // self._batch_size * self._batch_size

    def __repr__(self):
        return self.__class__.__name__ + f"({len(self)})"

    def __init__(self,
                 root: Optional[str] = None,
                 split: Literal["train", "test", "valid"] = "train",
                 num_classes: int = 1000,
                 sizes: Dict[str, int] = None,
                 max_tokens: int = 4096,
                 max_chunk_size: int = 1000,
                 processed_dir: str = "processed-data",
                 num_workers: int = 0,
                 mask_ratio: float = 0.5,
                 batch_size: int = 4,
                 **kwargs):
        if isinstance(root, str):
            root = os.path.expanduser(os.path.normpath(root))

        self._root = root
        self._split = split
        self._num_classes = num_classes
        self._processed_dir = processed_dir
        self._data_mapping = {}
        self._sample_mapping = {}
        self._size = sizes.get(split, 1)
        self._max_tokens = max_tokens
        self._mask_ratio = mask_ratio
        self._weights = None
        self._max_chunk_size = max_chunk_size
        self._with_samples = split == "test"
        self._shuffle = False
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._rank = get_global_rank()
        self._world_size = get_global_world_size()

    @property
    def raw_dir(self) -> str:
        return os.path.join(self._root, 'tensorised-data', self._split)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self._root, self._processed_dir, self._split)

    @property
    def counter_path(self):
        return os.path.join(self.processed_dir, "counter.pkl.gz")

    @property
    def sizes_path(self):
        return os.path.join(self.processed_dir, "sizes.pkl.gz")

    @property
    def indexes_path(self):
        return os.path.join(self.processed_dir, "indexes.pkl.gz")

    @property
    def processed_file_names(self) -> List[os.PathLike]:
        processed_path = pathlib.Path(self.processed_dir)
        return list(processed_path.glob("data_chunk_*.pkl.gz"))

    @property
    def counter(self) -> torch.Tensor:
        if not os.path.exists(self.counter_path):
            return torch.zeros(self._num_classes)
        return RichPath.create(self.counter_path).read_by_file_suffix()

    @property
    def sizes(self) -> Optional[torch.Tensor]:
        if not os.path.exists(self.sizes_path):
            return torch.tensor([0])
        return RichPath.create(self.sizes_path).read_by_file_suffix()

    @property
    def indexes(self) -> torch.Tensor:
        if not os.path.exists(self.sizes_path):
            return torch.arange(self._len)
        return RichPath.create(self.indxes_path).read_by_file_suffix()

    def process(self):
        if len(self.processed_file_names) == self._size:
            return self
        idx = 0
        indexes = []
        sizes = []

        counter = self.counter

        path = RichPath.create(self.raw_dir)
        paths = path.get_filtered_files_in_dir("chunk_*")

        result_path = RichPath.create(self.processed_dir)

        with ChunkWriter(result_path, file_prefix="data_chunk_", max_chunk_size=1000, file_suffix=".pkl.gz") as data_writer,\
             ChunkWriter(result_path, file_prefix="sample_chunk_", max_chunk_size=1000, file_suffix=".pkl.gz") as sample_writer:
            for chunk in tqdm(read_data_chunks(paths, num_workers=self._num_workers, max_queue_size=25), desc="Processing chunk"):
                for sample in tqdm(chunk, desc="Processing data"):
                    if "raw_data" not in sample and "supernodes" not in sample["raw_data"]:
                        continue
                    if np.all(sample["variable_target_class"] == 0):
                        continue
                    graph = sample_to_nx(sample)
                    if len(graph.graph["supernodes"]) == 0:
                        continue
                    data = nx_to_data(graph)
                    data.idx = idx

                    data_writer.add(data)
                    sample_writer.add(graph.graph)
                    values, counts = torch.unique(data.y, return_counts=True)
                    counter[values] += counts
                    sizes.append(data.y.size(0) + data.edge_index.size(1))
                    idx += 1
                    indexes.append(idx)

        # save indexes
        RichPath.create(self.indexes_path).save_as_compressed_file(torch.tensor(indexes))
        # save counter
        RichPath.create(self.counter_path).save_as_compressed_file(counter)
        # save sizes
        RichPath.create(self.sizes_path).save_as_compressed_file(torch.tensor(sizes))

        return self

    def _data_generator(self) -> Iterable[Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]]]]]:
        path = RichPath.create(self.processed_dir)
        data_paths = path.get_filtered_files_in_dir("data_chunk_*")
        idxs = list(range(len(data_paths)))

        mod = self._world_size
        shift = self._rank

        if self._shuffle:
            np.random.shuffle(idxs)

        def data_iter():
            data_paths = path.get_filtered_files_in_dir("data_chunk_*")
            data_paths = [data_paths[i] for i in idxs]

            for chunk in read_data_chunks(data_paths, num_workers=self._num_workers, max_queue_size=25):
                i = 0
                for data in chunk:
                    if filter_data(data, self._max_tokens):
                        continue
                    if (i + shift) % mod != 0:
                        continue
                    i += 1

                    yield preprocess_item(data, self._mask_ratio)

        def data_with_sample_iter():
            data_paths = path.get_filtered_files_in_dir("data_chunk_*")
            sample_paths = path.get_filtered_files_in_dir("sample_chunk_*")
            data_paths = [data_paths[i] for i in idxs]
            sample_paths = [sample_paths[i] for i in idxs]

            for data_chunk, sample_chunk in zip(
                    read_data_chunks(data_paths, num_workers=self._num_workers, max_queue_size=25),
                    read_data_chunks(sample_paths, num_workers=self._num_workers, max_queue_size=25)
            ):
                j = 0
                for i, data in enumerate(data_chunk):
                    if filter_data(data, self._max_tokens):
                        continue
                    if (j + shift) % mod != 0:
                        continue
                    j += 1

                    yield preprocess_item(data, self._mask_ratio), sample_chunk[i]

        if self._with_samples:
            return data_with_sample_iter()

        return data_iter()

    def __iter__(self):
        return iter(self._data_generator())

    def collater(self, items):
        if self._with_samples:
            data = (i for i, _ in items)
            sample = list(j for _, j in items)
            return collator(data), sample

        return collator(items)
