import tensorflow as tf
import gdown
import os

def checkGPUavailable():
    return True if len(tf.config.list_physical_devices('GPU')) else False


def download_weights(path, url):
    if not os.path.exists(path):
        os.mkdir('/'.join(path.split("/")[:-1]))
        print("Hang Tight Downloading the weights for the model")
        gdown.download(url=url, output=path, quiet=False, fuzzy=True)