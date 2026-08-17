import json
from pathlib import Path
from huggingface_hub import HfApi


registry_file = Path(__file__).parent / "model_registry.json"
with open(registry_file, "r") as f:
    registry = json.load(f)

repo_id = "naapeli/chess-puzzle-generator"
api = HfApi()

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

for revision, info in registry[repo_id].items():
    folder_path = info["export_folder"]

    if revision != "main":
        api.create_branch(repo_id=repo_id, branch=revision, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        revision=revision,
        repo_type="model",
        commit_message=f"Release {revision}",
    )
