import os
import json

config = {}

def read_config():
    # Get the path to the current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the config.json file
    config_path = os.path.join(script_dir, 'config.json')

    # Read the JSON file and convert it to a Python dictionary
    with open(config_path, 'r') as file:
        global config
        config = json.load(file)
        
    print(config)