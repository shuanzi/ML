import io
from os import listdir
from os.path import isfile, join, isdir


def _open(f):
    try:
        with io.open(f, "r") as my_file:
            return my_file.read()
    except BaseException as error:
        print("open file error, file: " + f)
        print(error)
        return ""

def get_all_file_path(dir):
    list = []
    for f in listdir(dir):
        if ".DS_Store" == f:
            continue
        filePath = join(dir, f)
        if isfile(filePath):
            list.append(filePath)
        elif isdir(filePath):
            list.extend(get_all_file_path(filePath))
    return list


def get_all_dir(dir):
    list = []
    for f in listdir(dir):
        dirPath = join(dir, f)
        if isdir(dirPath):
            list.append(f)
    return list
