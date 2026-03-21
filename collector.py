import pandas as pd
from .utils import now_tag, dump_json

class DataCollector:
    def __init__(self, main_data_file, raw_dir):
        self.main_data_file = main_data_file
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.current_row = 0
        self.processed_chunks = set()
        self.total_rows = self._get_total_rows()

    def get_next_batch(self, chunk_size = 10000):
        if not self.main_data_file.exists():
            return None, None

        try:
            if self.current_row >= self.total_rows:
                return None, None
            df = pd.read_csv(self.main_data_file, skiprows=range(1, self.current_row + 1), nrows=chunk_size)
            chunk_id = f"sequential_{self.current_row}_{self.current_row + chunk_size}"
            self.current_row += chunk_size

            if chunk_id in self.processed_chunks:
                return self.get_next_batch(chunk_size)

            self.processed_chunks.add(chunk_id)

            batch_id = now_tag()
            raw_path = self.raw_dir / f"batch_{batch_id}.csv"
            meta_path = self.raw_dir / f"batch_{batch_id}_meta.json"

            df.to_csv(raw_path, index=False)

            meta = {
                "batch_id": batch_id,
                "source_file": str(self.main_data_file),
                "raw_path": str(raw_path),
                "chunk_id": chunk_id,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "chunk_size": chunk_size,
                "current_row": self.current_row,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            dump_json(meta, meta_path)

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
