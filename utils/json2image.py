"""Script to convert data from json format into image-mask pair format."""
import os
import subprocess
from tqdm.auto import tqdm

from utils.data_utils import get_config


def json2image(json_path: str, sample_dir_path: str):
    """Run console commands to convert data from json
    into image-mask pair format using labelme tool.

    Args:
        json_path (str) -
        sample_dir_path (str) -
    """
    command = f"labelme_export_json {json_path} -o {sample_dir_path}"
    print(command)
    subprocess.run(command)


if __name__ == '__main__':
    path_to_json = "D:\\Hepatocyte\\7939_20_310320201319_7\\json\\grouped_full.json"
    path_to_samples = "D:\\Hepatocyte\\7939_20_310320201319_7\\full_sample"
    json2image(path_to_json, path_to_samples)
