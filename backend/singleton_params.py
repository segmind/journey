import os
import json

from main import singleton as gs

def instantiate_singleton_params():
    if not hasattr(gs, "data"):
        gs.data = {}
        gs.data["models"] = {}
        gs.base_loaded = None
        gs.refiner_loaded = None

def load_config_and_initialize():
    # Load the JSON file
    with open("config/defaults.json", "r") as f:
        config = json.load(f)
    # Create the default folders if they don't exist
    for key, folder in config['folders'].items():
        os.makedirs(folder, exist_ok=True)
    # Store the loaded configuration in gs.data
    if "config" not in gs.data:
        gs.data["config"] = {}
    gs.data["config"].update(config)
    return config

instantiate_singleton_params()

global config
config = load_config_and_initialize()

for name, path in config["folders"].items():
    os.makedirs(path, exist_ok=True)

