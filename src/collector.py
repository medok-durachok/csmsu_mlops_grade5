import pandas as pd
from pathlib import Path
from .utils import now_tag, dump_json
from .config_loader import DataCollectionConfigLoader

class DataCollector:
    def __init__(self, main_data_file=None, raw_dir=None, config_path="data_collection_config.yaml"):
        try:
            self.dc_config = DataCollectionConfigLoader(config_path)
            self.use_config = True
        except (FileNotFoundError, ValueError):
            self.dc_config = None
            self.use_config = False

        if self.use_config and self.dc_config:
            self.main_data_file = Path(self.dc_config.main_file) if main_data_file is None else Path(main_data_file)
            self.raw_dir = Path(self.dc_config.raw_dir) if raw_dir is None else Path(raw_dir)
            self.batch_size = self.dc_config.batch_size
            self.collection_mode = self.dc_config.collection_mode
            self.random_seed = self.dc_config.random_seed
            self.log_progress = self.dc_config.log_progress
            self.save_metadata = self.dc_config.save_metadata
        else:
            self.main_data_file = Path(main_data_file) if main_data_file else Path("motor_data14-2018.csv")
            self.raw_dir = Path(raw_dir) if raw_dir else Path("data/raw")
            self.batch_size = 10000
            self.collection_mode = "sequential"
            self.random_seed = 42
            self.log_progress = True
            self.save_metadata = True

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.current_row = 0
        self.processed_chunks = set()
        self.total_rows = self._get_total_rows()
        self.batches_collected = 0

    def get_next_batch(self, chunk_size=None):
        chunk_size = chunk_size or self.batch_size
        if not self.main_data_file.exists():
            return None, None

        try:
            if self.current_row >= self.total_rows:
                return None, None

            if self.collection_mode == "random":
                df_full = pd.read_csv(self.main_data_file)
                df = df_full.sample(n=min(chunk_size, len(df_full)), random_state=self.random_seed + self.batches_collected)
                chunk_id = f"random_{self.batches_collected}_{chunk_size}"
            else:
                df = pd.read_csv(self.main_data_file, skiprows=range(1, self.current_row + 1), nrows=chunk_size)
                chunk_id = f"sequential_{self.current_row}_{self.current_row + chunk_size}"
                self.current_row += chunk_size

            if chunk_id in self.processed_chunks:
                return self.get_next_batch(chunk_size)

            self.processed_chunks.add(chunk_id)
            self.batches_collected += 1

            batch_id = now_tag()
            raw_path = self.raw_dir / f"batch_{batch_id}.csv"
            meta_path = self.raw_dir / f"batch_{batch_id}_meta.json"

            df.to_csv(raw_path, index=False)

            meta = {
                "batch_id": batch_id,
                "source_file": str(self.main_data_file),
                "raw_path": str(raw_path),
                "chunk_id": chunk_id,
                "collection_mode": self.collection_mode,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "chunk_size": chunk_size,
                "current_row": self.current_row,
                "batches_collected": self.batches_collected,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }

            if self.save_metadata:
                dump_json(meta, meta_path)

            if self.log_progress and self.batches_collected % 1 == 0:
                print(f"Collected batch {self.batches_collected}: {len(df)} rows")

            return df, meta

        except Exception as e:
            print(f"Error reading data file: {e}")
            return None, None

    def _get_total_rows(self):
        if not self.main_data_file.exists():
            return 0
        try:
            with open(self.main_data_file, 'r') as f:
                return sum(1 for _ in f) - 1
        except:
            return 0

    def reset(self):
        self.current_row = 0
        self.processed_chunks.clear()
        self.batches_collected = 0

    def get_status(self):
        return {
            "total_rows": self.total_rows,
            "current_row": self.current_row,
            "batches_collected": self.batches_collected,
            "collection_mode": self.collection_mode,
            "batch_size": self.batch_size,
            "progress_percent": (self.current_row / self.total_rows * 100) if self.total_rows > 0 else 0
        }
