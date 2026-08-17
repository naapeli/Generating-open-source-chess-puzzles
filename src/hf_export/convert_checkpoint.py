import sys
import json
import shutil
import pickle
from pathlib import Path
import torch

from hf_export.model import MaskedDiffusion as HFMaskedDiffusion
from hf_export.pipeline import ChessPuzzlePipeline


class DummyConfig:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.__dict__.update(state[1])

class DummySchedule:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class DummyFallback:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect missing class modules to dummy stubs
        if module == "Config" or module.endswith("Config"):
            return DummyConfig
        if "MaskingSchedule" in module:
            return DummySchedule
        if "tokenization" in module:
            return DummyFallback
        try:
            return super().find_class(module, name)
        except Exception:
            return DummyFallback

class SafePickleModule:
    Unpickler = SafeUnpickler
    
    @staticmethod
    def load(f, **kwargs):
        encoding = kwargs.get("encoding", "ASCII")
        return SafeUnpickler(f, encoding=encoding).load()

    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL

# ==========================================================
# Checkpoint Conversion
# ==========================================================

def convert(checkpoint_path, output_dir):
    checkpoint_file = Path(checkpoint_path)
    output_path = Path(output_dir)

    print(f"Loading checkpoint from: {checkpoint_file}...")
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file '{checkpoint_file}' not found.")
        return

    # Load using our SafePickleModule to bypass missing dependency errors
    checkpoint = torch.load(checkpoint_file, map_location="cpu", pickle_module=SafePickleModule)
    local_config = checkpoint.get("config", DummyConfig())

    print("Mapping model parameters and configuration...")
    hf_model = HFMaskedDiffusion(
        n_fen_tokens=getattr(local_config, "n_fen_tokens", 48),
        n_move_tokens=getattr(local_config, "n_move_tokens", 4),
        n_themes=getattr(local_config, "n_themes", 66),
        rating_dim=getattr(local_config, "rating_dim", 1),
        fen_length=getattr(local_config, "fen_length", 76),
        move_length=getattr(local_config, "move_length", 5),
        predict_moves=getattr(local_config, "predict_moves", True),
        use_context=getattr(local_config, "use_context", True),
        n_heads=getattr(local_config, "n_heads", 8),
        n_layers=getattr(local_config, "n_layers", 16),
        embed_dim=getattr(local_config, "embed_dim", 1024)
    )

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    
    # Strip compile prefixes like '_orig_mod.' if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod."):]] = v
        else:
            clean_state_dict[k] = v

    hf_model.load_state_dict(clean_state_dict)
    print("Weights loaded successfully!")

    # Instantiate custom pipeline
    pipeline = ChessPuzzlePipeline(model=hf_model)

    # Save pretrained components to output folder
    print(f"Saving Diffusers pretrained format to: {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(output_path)

    # Copy pipeline.py to root and model.py to model/ for Diffusers custom module loader
    current_dir = Path(__file__).resolve().parent
    shutil.copy(current_dir / "pipeline.py", output_path / "pipeline.py")
    (output_path / "model").mkdir(parents=True, exist_ok=True)
    shutil.copy(current_dir / "model.py", output_path / "model" / "model.py")

    # Inject module class references into model_index.json
    model_index_path = output_path / "model_index.json"
    if model_index_path.exists():
        with open(model_index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        index_data.pop("auto_map", None)
        # Point class references to exported standalone files (pipeline.py and model/model.py)
        index_data["_class_name"] = ["pipeline", "ChessPuzzlePipeline"]
        index_data["model"] = ["model", "MaskedDiffusion"]
        with open(model_index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    print("\nConversion successfully completed!")
    print(f"Export directory '{output_path}' is now ready to upload to Hugging Face Hub.")
    print("Files ready to be published:")
    for file_path in sorted(output_path.rglob("*")):
        if file_path.is_file():
            print(f" - {file_path.relative_to(output_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run --directory src python -m hf_export.convert_checkpoint <path_to_checkpoint.pt> [output_directory]")
        print("Example: uv run --directory src python -m hf_export.convert_checkpoint ./runs/rl/final_large_runs/ownThemeDistribution/run10/model_0020000.pt ./test_hf_export")
        sys.exit(1)
    
    chk_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./packaged_pipeline"
    convert(chk_path, out_dir)
