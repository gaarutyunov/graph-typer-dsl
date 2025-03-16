from collections import namedtuple
from typing import List, Dict, Any, Iterable, Optional

import numpy as np
from dpu_utils.utils import RichPath, MultiWorkerCallableIterator

ModelTestResult = namedtuple("ModelTestResult", ["ground_truth", "all_predictions"])


NONE_TOKEN = '<NONE>'


def get_data_files_from_directory(data_dir: RichPath, max_num_files: Optional[int]=None) -> List[RichPath]:
    files = data_dir.get_filtered_files_in_dir('*.gz')
    if max_num_files is None:
        return files
    else:
        return sorted(files)[:int(max_num_files)]


def read_data_chunks(data_chunk_paths: Iterable[RichPath], shuffle_chunks: bool=False, max_queue_size: int=1, num_workers: int=0) \
        -> Iterable[List[Dict[str, Any]]]:
    if shuffle_chunks:
        data_chunk_paths = list(data_chunk_paths)
        np.random.shuffle(data_chunk_paths)
    if num_workers <= 0:
        for data_chunk_path in data_chunk_paths:
            yield data_chunk_path.read_by_file_suffix()
    else:
        def read_chunk(data_chunk_path: RichPath):
            return data_chunk_path.read_by_file_suffix()
        yield from MultiWorkerCallableIterator(argument_iterator=[(data_chunk_path,) for data_chunk_path in data_chunk_paths],
                                               worker_callable=read_chunk,
                                               max_queue_size=max_queue_size,
                                               num_workers=num_workers,
                                               use_threads=True)