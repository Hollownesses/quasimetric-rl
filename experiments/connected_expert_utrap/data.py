"""Training adapter; graph references are never loaded here."""
import json
from pathlib import Path
from experiments.expert_utrap_network.training_adapter import (
    load_training_data as load_backbone_format, audit_training_data)


def load_training_data(root, *, task_mode='all'):
    manifest=json.loads((Path(root)/'manifest.json').read_text())
    if manifest.get('dataset_kind')!='connected_expert_network':
        raise ValueError('expected connected_expert_network dataset')
    data=load_backbone_format(root,task_mode=task_mode)
    data.dataset_kind='connected_expert_network'
    data.fingerprint.update(training_dataset=data.dataset_kind,
                            training_sampling='single_source_connected_expert_records_v1')
    return data
