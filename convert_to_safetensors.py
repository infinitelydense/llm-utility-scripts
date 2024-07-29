import torch
import os
import glob
from safetensors.torch import save_file

# Configuration variable
INPUT_DIR = "./input_models"  # Set this to your desired input directory

def convert_to_safetensors(input_dir, unshare=False):
    """
    Convert .bin/.pt files in the specified directory to .safetensors format.
    
    Args:
    input_dir (str): Path to the directory containing .bin/.pt files
    unshare (bool): If True, detach tensors to prevent any from sharing memory
    """
    tensor_files = glob.glob(os.path.join(input_dir, "*.bin")) + glob.glob(os.path.join(input_dir, "*.pt"))
    
    for file in tensor_files:
        print(f" -- Loading {file}...")
        state_dict = torch.load(file, map_location="cpu")

        if unshare:
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].clone().detach()

        out_file = os.path.splitext(file)[0] + ".safetensors"
        print(f" -- Saving {out_file}...")
        save_file(state_dict, out_file, metadata={"format": "pt"})

    print("Conversion completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert .bin/.pt files to .safetensors")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR, help="Input directory containing .bin/.pt files")
    parser.add_argument("--unshare", action="store_true", help="Detach tensors to prevent any from sharing memory")
    args = parser.parse_args()

    convert_to_safetensors(args.input_dir, args.unshare)