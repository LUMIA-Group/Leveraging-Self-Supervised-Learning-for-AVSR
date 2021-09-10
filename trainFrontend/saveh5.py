import cv2 as cv
import dlib
import h5py
import numpy as np
from tqdm import tqdm

from config import args
from preprocess.find_mean import find_mean_face
from preprocess.preprocessing import preprocessing
from trainFrontend.generate_filelist import generate_filelist


def get_files(datadir, dataset):
    with open(datadir + "/" + dataset + ".txt", "r") as f:
        datalist = f.readlines()
    datalist = [line.strip() for line in datalist]
    return datalist


def fill_in_data(datalist, png):
    for i, item in enumerate(tqdm(datalist, leave=False, desc="saveh5", ncols=75)):
        vidInp = cv.imread(item + '.png')
        vidInp = cv.cvtColor(vidInp, cv.COLOR_BGR2GRAY)
        vidInp = cv.imencode(".png", vidInp)[1].tobytes()
        png[i] = np.frombuffer(vidInp, np.uint8)


def main():
    """
        Preparation for model and filelist
    """
    generate_filelist()
    datadir = "../" + args['LRW_DATA_DIRECTORY']
    train_datalist = get_files(datadir, 'train')
    val_datalist = get_files(datadir, 'val')
    test_datalist = get_files(datadir, 'test')
    filesList = train_datalist + val_datalist + test_datalist

    landmark_detector = dlib.shape_predictor("../" + args["SHAPE_PREDICTOR_FILE"])
    mean_face_landmarks = find_mean_face(train_datalist, args["PREPROCESSING_NUM_OF_PROCESS"], landmark_detector)
    preprocessing(filesList, args["PREPROCESSING_NUM_OF_PROCESS"], landmark_detector, mean_face_landmarks, False, "lrw")

    """
        Create dataset and Load data
    """

    f = h5py.File("../" + args["LRW_HDF5_FILE"], "w")
    dt = h5py.vlen_dtype(np.dtype('uint8'))
    png = f.create_dataset('png', (len(filesList),), dtype=dt)
    fill_in_data(filesList, png)
    f.close()


if __name__ == "__main__":
    main()
