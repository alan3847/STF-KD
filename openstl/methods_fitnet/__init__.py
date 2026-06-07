from .fitnet_kd_method import FitNetKDMethod

method_maps = {
    'fitnet_kd': FitNetKDMethod,
}

__all__ = [
    'method_maps',
    'FitNetKDMethod',
]
