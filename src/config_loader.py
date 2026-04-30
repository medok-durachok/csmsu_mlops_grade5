import yaml
from pathlib import Path
from typing import Dict, Any


class DataCollectionConfigLoader:
    def __init__(self, config_path = "data_collection_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)

        if 'data_collection' not in full_config:
            raise ValueError("Configuration must contain 'data_collection' section")

        return full_config['data_collection']

    def get(self, key_path, default = None):
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    @property
    def batch_size(self):
        return self.get('batching.batch_size', 10000)

    @property
    def collection_mode(self):
        return self.get('strategy.mode', 'sequential')

    @property
    def random_seed(self):
        return self.get('strategy.random_seed', 42)

    @property
    def main_file(self):
        return self.get('source.main_file', 'motor_data14-2018.csv')

    @property
    def raw_dir(self):
        return self.get('storage.raw_dir', 'data/raw')

    @property
    def save_metadata(self):
        return self.get('metadata.save_metadata', True)

    @property
    def log_progress(self):
        return self.get('monitoring.log_progress', True)
