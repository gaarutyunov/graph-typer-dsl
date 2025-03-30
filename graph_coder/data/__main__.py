import argparse
import logging
from multiprocessing.pool import ApplyResult
from pathlib import Path
from typing import List
from dpu_utils.utils import RichPath
from multiprocessing import Pool, cpu_count

from graph_coder.data.process import process_file


def main(args: argparse.Namespace) -> None:
    resolved_root = Path(args.dataset_root).expanduser()
    resolved_path = resolved_root / "tensorised-data" / args.split
    data_path = RichPath.create(str(resolved_path))

    output_path = resolved_root / args.processed_dir / args.split
    output_path.mkdir(parents=True, exist_ok=True)

    output_rich_path = RichPath.create(str(output_path))

    i = 0

    logging.info(f"Processing data from {data_path} to {output_rich_path}")
    logging.info(f"Max tokens: {args.max_tokens}")
    logging.info(f"Mask: {args.mask}")
    logging.info(f"Number of workers: {args.num_workers}")

    if args.num_workers <= 1:
        logging.info("Using single process")
        for file_path in data_path.iterate_filtered_files_in_dir("chunk_*.pkl.gz"):
            output_file_path = output_rich_path.join(f"data_chunk_{i}.pkl.gz")
            process_file(file_path, output_file_path, args.mask, args.max_tokens, i)
            i += 1

        return

    with Pool(args.num_workers) as pool:
        jobs: List[ApplyResult] = []

        for i, file_path in enumerate(data_path.iterate_filtered_files_in_dir("chunk_*.pkl.gz")):
            output_file_path = output_rich_path.join(f"data_chunk_{i}.pkl.gz")
            jobs.append(
                pool.apply_async(process_file, args=(file_path, output_file_path, args.mask, args.max_tokens, i))
            )
        
        for job in jobs:
            job.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="~/git-py", help="Dataset root path")
    parser.add_argument("--split", choices=["train", "valid", "test"], type=str, default="train")
    parser.add_argument("--max-tokens", default=10000, type=int, help="Max number of tokens (nodes and edges) in graph")
    parser.add_argument("--processed-dir", default="processed-data", help="Processed data directory")
    parser.add_argument("--mask", default=False, action="store_true", help="Whether to mask tokens")
    parser.add_argument("--num-workers", default=cpu_count(), type=int, help="Number of workers to use for data loading")

    args = parser.parse_args()
    main(args)
