import argparse
import os
import pickle


parser = argparse.ArgumentParser(description="Load a file path from command line arguments.")
parser.add_argument('--file', type=str, help='Path to the file', required=True)
args = parser.parse_args()
file_path = args.file

if not os.path.isfile(file_path):
    raise ValueError(f"The file {file_path} does not exist.")

with open(file_path, 'rb') as file:
    model_dict = pickle.load(file)

print(model_dict)
