from dataclasses import dataclass, field
from fairseq.tasks import FairseqTask, register_task
from graph_coder.data.registry import build_dataset
from tokengt.tasks.graph_prediction import GraphPredictionConfig


@dataclass
class DeepSimilarityLearningConfig(GraphPredictionConfig):
    dataset_name: str = field(
        default="typilus",
        metadata={"help": "name of the dataset"},
    )

    dataset_root: str = field(
        default="~/data",
        metadata={"help": "Dataset root folder"},
    )

    processed_dir: str = field(
        default="processed-dir",
        metadata={"help": "Dataset processed folder"},
    )

    num_data_workers: int = field(
        default=4,
        metadata={"help": "number of data workers"},
    )

    max_nodes: int = field(
        default=10000,
        metadata={"help": "number nodes per graph"},
    )

    num_atoms: int = field(
        default=10000 + 512 + 2,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=8 + 2,
        metadata={"help": "number of edge types in the graph"},
    )

    num_classes: int = field(
        default=100,
        metadata={"help": "number of classes"},
    )


@register_task("deep_similarity_learning", dataclass=DeepSimilarityLearningConfig)
class DeepSimilarityLearningTask(FairseqTask):
    def load_dataset(self, split, combine = False, task_cfg = None, **kwargs):
        self.datasets[split] = build_dataset(self.cfg.dataset_name, self.cfg, split, **kwargs)
    
    @property
    def source_dictionary(self):
        return None
    
    @property
    def target_dictionary(self):
        return None
