from .fakd_kd_method import FrequencyAlignedKDMethod

method_maps = {
    'fakd_kd': FrequencyAlignedKDMethod,
}

__all__ = [
    'method_maps',
    'FrequencyAlignedKDMethod',
]
