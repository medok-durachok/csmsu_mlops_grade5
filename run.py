import argparse
import sys
from src.model_storage import ModelStorage
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import CFG
from src.pipeline import MLPipeline
from src.utils import ensure_dirs


def setup_directories():
    ensure_dirs(
        CFG.RAW_DIR,
        CFG.PROCESSED_DIR,
        CFG.MODELS_DIR,
        CFG.OUTPUTS_DIR,
        CFG.REPORTS_DIR
    )

def mode_update(args):
    print("UPDATE MODE")
    pipeline = MLPipeline(CFG)

    if args.chunk_size:
        chunk_size = int(args.chunk_size)
    else:
        chunk_size = CFG.BATCH_SIZE

    try:
        if pipeline.is_trained:
            print("Updating existing model: ")
            result = pipeline.process_batch()
        else:
            print("Training initial mode:")
            result = pipeline.train_with_real_data(chunk_size=chunk_size)

        if result:
            print(f"Success: {result}")
        else:
            print("No data available for processing")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def mode_summary():
    print("SUMMARY MODE")
    pipeline = MLPipeline(CFG)

    if pipeline.is_trained:
        print("Pipeline is trained")
        print(f"Current model version: {pipeline.current_model_version}")

        try:
            _, _, metadata = pipeline.load_model()
            print(f"  Model Name: {metadata.get('model_name', 'unknown')}")
            print(f"  Version ID: {metadata.get('version_id', 'unknown')}")
            print()

            if 'metrics' in metadata:
                print("Model Metrics:")
                metrics = metadata['metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        print(f"  {metric_name}: {metric_value:.4f}")
                    else:
                        print(f"  {metric_name}: {metric_value}")
                print()
        except Exception as e:
            print(f"Could not load current model metadata: {e}")
    else:
        print("Pipeline is not trained yet")
    print()

    try:
        model_storage = ModelStorage(storage_dir=str(CFG.MODELS_DIR))
        if model_storage.metadata and model_storage.metadata.get("models"):
            models_info = model_storage.metadata["models"]
            for version_id in sorted(models_info.keys(), reverse=True):
                info = models_info[version_id]
                print(f"\nModel: {version_id}")

                if 'metrics' in info:
                    metrics = info['metrics']
                    print(f"  Metrics:")
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, float):
                            print(f"    {metric_name}: {metric_value:.4f}")
                        else:
                            print(f"    {metric_name}: {metric_value}")
        else:
            print("No trained models found in storage")
    except Exception as e:
        print(f"Error checking models: {e}")


def main():
    setup_directories()

    parser = argparse.ArgumentParser(description="MLOPS System")
    parser.add_argument('-mode', required=True, choices=['update', 'summary', 'inference'], help='Operation mode')
    parser.add_argument('-chunk_size', help='Batch size for update mode')
    args = parser.parse_args()

    if args.mode == 'update':
        mode_update(args)
    elif args.mode == 'summary':
        mode_summary()
    elif args.mode == 'inference':
        pass
    else:
        print("Invalid mode")
        sys.exit(1)

if __name__ == "__main__":
    main()