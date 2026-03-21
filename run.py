import argparse
import sys
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


def mode_summary(args):
    print("SUMMARY MODE")
    pipeline = MLPipeline(CFG)

    if pipeline.is_trained:
        print("Pipeline is trained")
        print(f"Current model version: {pipeline.current_model_version}")

        try:
            _, _, metadata = pipeline.load_model()
            print(f"Model metadata: {metadata}")
        except:
            print("Could not load model metadata")
    else:
        print("Pipeline is not trained")


def main():
    setup_directories()

    parser = argparse.ArgumentParser(description="MLOPS System")
    parser.add_argument('-mode', required=True, choices=['update', 'summary', 'inference'], help='Operation mode')
    parser.add_argument('-chunk_size', help='Batch size for update mode')
    args = parser.parse_args()

    if args.mode == 'update':
        mode_update(args)
    elif args.mode == 'summary':
        mode_summary(args)
    elif args.mode == 'inference':
        pass
    else:
        print("Invalid mode")
        sys.exit(1)

if __name__ == "__main__":
    main()