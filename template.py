import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

folder_files_list = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    "requirements.txt",
    "setup.py",
    "research/01_Vcc_Trials.ipynb",
    "templates/index.html",
    "artifacts/01_Data/delete.csv",
    "artifacts/02_DataFrames/delete.csv",
    "artifacts/03_Pipeline/delete.csv",
    "artifacts/04_Models/delete.csv",
    "artifacts/05_Scores/delete.csv",
    "logs/delete.csv",
]

for path in folder_files_list:
    file_path = Path(path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'Created directory "{file_dir}" for file "{file_name}"')

    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w") as f:
            pass
        logging.info(f'Created empty file: "{path}"')
    else:
        logging.info(f'"{path}" already exists')
