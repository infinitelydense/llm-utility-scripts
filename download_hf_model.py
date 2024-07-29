import os
import re
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# Config variables
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
BRANCH_NAME = "main"
OUTPUT_DIR = "./models"

def sanitize_model_and_branch_names(model, branch):
    # Remove trailing slash if present
    model = model.rstrip('/')

    # Remove base URL if present
    if model.startswith("https://huggingface.co/"):
        model = model[len("https://huggingface.co/"):]

    # Split model and branch if provided in model name
    model_parts = model.split(":")
    model = model_parts[0]
    branch = model_parts[1] if len(model_parts) > 1 else branch

    # Use 'main' as default branch if not specified
    if branch is None:
        branch = "main"

    # Validate branch name
    if not re.match(r"^[a-zA-Z0-9._-]+$", branch):
        raise ValueError("Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

    return model, branch

def download_model(model_name, branch_name, output_dir):
    # Sanitize model and branch names
    model_name, branch_name = sanitize_model_and_branch_names(model_name, branch_name)

    # Initialize Hugging Face API
    api = HfApi()

    # Create output directory
    output_folder = Path(output_dir) / f"{'_'.join(model_name.split('/')[-2:])}"
    if branch_name != "main":
        output_folder = output_folder.with_name(f"{output_folder.name}_{branch_name}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get file list
    try:
        files = api.list_repo_files(model_name, revision=branch_name)
    except Exception as e:
        print(f"Error accessing repository: {e}")
        return

    # Download files
    for file in tqdm(files, desc="Downloading files"):
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=file,
                revision=branch_name,
                local_dir=output_folder,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Error downloading {file}: {e}")

    print(f"Model downloaded to {output_folder}")

if __name__ == "__main__":
    try:
        download_model(MODEL_NAME, BRANCH_NAME, OUTPUT_DIR)
    except Exception as e:
        print(f"An error occurred: {e}")