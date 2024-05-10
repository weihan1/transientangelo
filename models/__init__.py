models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model


from . import geometry, texture, transient_nerf, transient_neus, captured_nerf, captured_neus, baseline_neus, baseline_neus_captured
