import json

import os
def load_file(path):
    """

    :param path:path to json file
    :return: data
    """
    with open(path) as f:
        return json.load(f)


def save_file(data,path,indent=0):
    """

    :param vocab: data
    :param path: path to save
    :param indent: size of indent
    :return:
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile,indent=indent)

def create_directory(path):

    if os.path.isdir(path):
        pass
    else:
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
