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
    """
        Arg filename comes in format: '/home/user/yalefaces/subject03.sad.gif',
        with subject ids starting in 01 and ending in 11.

        This function first gets the file name without extension,
            result: 'subject03.sad'
        then gets the subject identifier,
            result: 'subject03'
        then gets the number in its identifier,
            result: 03
        then converts to integer and subtracts 1 so our labels stay in range [0, 10]
            result: 2
    """

    filename = os.path.split(filepath)[1]
    subject = filename.split(".")[0]
    subject_id = subject.replace("subject", "")
    return int(subject_id) - 1

def load(folder_path: str):
    files = _get_filepaths(folder_path)

    X = np.array(list(map(_to_image, files)))
    y = np.array(list(map(_to_label, files)))

    return X, y