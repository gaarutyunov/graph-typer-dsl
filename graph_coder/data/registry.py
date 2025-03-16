DATASET_REGISTRY = {}

def register_dataset(name):
    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(
                "Cannot register duplicate dataset ({})".format(name)
            )
        if cls.__name__ in DATASET_REGISTRY:
            raise ValueError(
                "Cannot register dataset with duplicate class name ({})".format(
                    cls.__name__
                )
            )

        DATASET_REGISTRY[name] = cls

        return cls

    return register_dataset_cls


def build_dataset(name, cfg, *args, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(
            "Unknown dataset architecture: {}".format(name)
        )
    return DATASET_REGISTRY[name](cfg, *args, **kwargs)


__all__ = [
    "build_dataset",
    "register_dataset",
]