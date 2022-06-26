import os
import numpy as np
from PIL import Image

SHAPE=(243, 320)

def _get_filepaths(folder_path: str):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

def _to_image(filepath: str):
    image_pil = Image.open(filepath)
    return np.array(image_pil, 'uint8').flatten()

def _to_label(filepath: str):
    # "/home/yalefaces/subject03.sad.gif" -> "subject03.sad"
    filename = os.path.split(filepath)[1]

    # "subject03.sad" -> "subject03"
    subject = filename.split(".")[0]

    # "subject03" -> "03"
    subject_id = subject.replace("subject", "")

    # "03" -> 2
    return int(subject_id) - 1

def load(folder_path: str):
    files = _get_filepaths(folder_path)

    X = np.array(list(map(_to_image, files)))
    y = np.array(list(map(_to_label, files)))

    return X, y
