import argparse
import yaml
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script with external config')
    parser.add_argument('--config', help='Path to the config file', required=True)
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The config file {config_path} does not exist.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)