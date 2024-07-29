import yaml
import os
import subprocess
import logging
from huggingface_hub import HfApi, create_repo

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            error_output = process.stderr.read()
            logging.error(f"Error executing command: {command}")
            logging.error(f"Error output: {error_output}")
            return False
        
        logging.info(f"Command executed successfully: {command}")
        return True
    except Exception as e:
        logging.error(f"Error executing command: {command}")
        logging.error(f"Error: {str(e)}")
        return False

def quantize_and_upload(config):
    exllama_path = os.path.expanduser(config['exllama_path'])
    base_model_name = config['base_model_name']
    input_model_path = config['input_model_path']
    output_base_path = config['output_base_path']
    hf_username = config['hf_username']
    default_hb = config.get('default_hb', 8)

    for quant_config in config['quantizations']:
        bpw = quant_config['bpw']
        hb = quant_config.get('hb', default_hb)
        
        quant_name = f"{base_model_name}-exl2-{bpw}bpw"
        work_dir = os.path.join(output_base_path, base_model_name, f"{quant_name}-work")
        output_dir = os.path.join(output_base_path, base_model_name, quant_name)

        try:
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories for {quant_name}: {str(e)}")
            continue

        # Run quantization
        command = f"python {exllama_path}/convert.py -i {input_model_path} -o {work_dir} -cf {output_dir} -b {bpw} -hb {hb}"
        if not run_command(command):
            logging.error(f"Quantization failed for {quant_name}. Skipping upload.")
            continue

        logging.info(f"Quantization completed for {quant_name}")

        # Try to upload to Hugging Face
        repo_name = f"{hf_username}/{quant_name}"
        try:
            create_repo(repo_name, repo_type="model", exist_ok=True)
            api = HfApi()
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            logging.info(f"Successfully uploaded {quant_name} to Hugging Face")
        except Exception as e:
            logging.error(f"Failed to upload {quant_name} to Hugging Face: {str(e)}")
            logging.info(f"Quantized model is still available locally at {output_dir}")

        logging.info(f"Completed processing for {quant_name}")

if __name__ == "__main__":
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error reading config.yaml: {str(e)}")
        exit(1)
    except FileNotFoundError:
        logging.error("config.yaml not found. Please create a config file.")
        exit(1)

    quantize_and_upload(config)
    logging.info("Script execution completed.")