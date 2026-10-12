"""Graph-only reference sets paired with qm9s_ters_frozen.py."""
from configs.base_ters_qm9s import get_config as _base_config
from configs.qm9s_ters_frozen import QM9S_ROOT, QM9S_GRAPHS


def get_config():
    config = _base_config()
    config.data.root = QM9S_ROOT
    config.data.graph_root = QM9S_GRAPHS
    return config
