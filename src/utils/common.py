import os
import json
import base64
from pathlib import Path


def create_directories(directories_path: list, verbose=True):
    """creates list of directories
       
       Args:
            directories_path (list): list of directories path
            verbose (bool): status to show the directory creation log
    
    """
    for directory_path in directories_path:
        os.makedirs(directory_path, exist_ok=True)
        if verbose:
            pass


def encodeImageIntoBase64(img_path: Path):

    """return image as base64 encoded string

       Args:
            img_path (Path): path of the image
    
       Returns:
            str: base64 encoded string
    """
    with open(img_path, 'rb') as f:
        img_content = f.read()
        return base64.b64encode(img_content)


def decodeImage(img_str: str, file_name: str):
    """save as image file from image string data
    
       Args:
            img_str (str): image data in string format
            file_name (str): path to save the image

    """
    img_data = base64.b64decode(img_str)
    with open(file_name, 'wb') as f:
        f.write(img_data)
            