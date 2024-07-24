datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import blender, colmap, dtu, transient_blender, captured_dataset, baseline_transient, baseline_captured_transient, captured_dataset_movie, blender_dataset_movie
