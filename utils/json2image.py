"""Script to convert data from json format into image-mask pair format."""
import os
import subprocess
from tqdm.auto import tqdm

from utils.data_utils import get_config


def json2voc(selected_dataset='GROUPED'):
    """
    Run console commands to convert data from json
    into image-mask pair format using labelme tool.

    Args:
        selected_dataset (str) - selected json dataset."""
    config = get_config(path='../config.yml')
    path_to_json = config['PATH']['JSON'][selected_dataset]
    path_to_images = config['PATH']['JSON']['IMAGE_MASK'][selected_dataset]
    file_names = os.listdir(path_to_json)
    commands = [
        f"labelme_export_json {path_to_json}{file_name} -o {path_to_images}{file_name[:file_name.rfind('.')]}"
        for file_name in file_names]
    for cmd in tqdm(commands):
        subprocess.check_output(cmd)


if __name__ == '__main__':
    dataset_to_convert = 'GROUPED'
    json2voc(dataset_to_convert)
