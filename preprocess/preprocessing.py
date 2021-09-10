import os
from collections import deque
from multiprocessing import Process

import cv2 as cv
import dlib
import numpy as np
from skimage import transform as tf
from tqdm import tqdm

STD_SIZE = (224, 224)
stablePntsIDs = [33, 36, 39, 42, 45]


def shape_to_array(shape):
    coords = np.empty((68, 2))
    for i in range(0, 68):
        coords[i][0] = shape.part(i).x
        coords[i][1] = shape.part(i).y
    return coords


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def crop_patch(frames, landmarks, mean_face_landmarks):
    """Crop mouth patch
    :param str frames: video_frames
    :param list landmarks: interpolated landmarks
    """

    for frame_idx, frame in enumerate(frames):
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == 12:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans = tf.estimate_transform('similarity', smoothed_landmarks[stablePntsIDs, :], mean_face_landmarks[stablePntsIDs, :])
            trans_frame = tf.warp(cur_frame, inverse_map=trans.inverse, output_shape=STD_SIZE)
            trans_frame = trans_frame * 255  # note output from wrap is double image (value range [0,1])
            trans_frame = trans_frame.astype('uint8')
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(cut_patch(trans_frame, trans_landmarks[48:68], 60, 60))
        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = tf.warp(cur_frame, inverse_map=trans.inverse, output_shape=STD_SIZE)
                trans_frame = trans_frame * 255  # note output from wrap is double image (value range [0,1])
                trans_frame = trans_frame.astype('uint8')
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(cut_patch(trans_frame, trans_landmarks[48:68], 60, 60))
            return np.array(sequence)
    return None


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def preprocess_sample(file, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop):
    """
    Function to preprocess each data sample.
    """

    videoFile = file + ".mp4"
    audioFile = file + ".flac"
    roiFile = file + ".png"

    # Extract the audio from the video file using the FFmpeg utility and save it to a flac file.
    if withaudio:
        v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
        os.system(v2aCommand)

    # for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    frames = list()
    landmarks = list()
    while captureObj.isOpened():
        ret, frame = captureObj.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if not len(frame) == 224:
                frame = cv.resize(frame, (224, 224))
            frames.append(frame)

            face_rects = face_detector(frame, 0)  # Detect face
            if len(face_rects) < 1:
                landmarks.append(None)
                continue
            rect = face_rects[0]  # Proper number of face
            landmark = landmark_detector(frame, rect)  # Detect face landmarks
            landmark = shape_to_array(landmark)
            landmarks.append(landmark)
        else:
            break
    captureObj.release()

    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if preprocessed_landmarks is None:
        if defaultcrop == "lrs":
            frames = [frame[52:172, 52:172] for frame in frames]
        else:
            frames = [frame[103: 223, 67: 187] for frame in frames]
    else:
        frames = crop_patch(frames, preprocessed_landmarks, mean_face_landmarks)

    assert frames is not None, "cannot crop from {}.".format(videoFile)

    cv.imwrite(roiFile, np.concatenate(frames, axis=1).astype(int))


def preprocess_sample_list(filesList, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop):
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop)


def preprocessing(filesList, processes, landmark_detector, mean_face_landmarks, withaudio, defaultcrop):
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")

    face_detector = dlib.get_frontal_face_detector()

    def splitlist(inlist, chunksize):
        return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]

    filesListSplitted = splitlist(filesList, int((len(filesList) / processes)))

    process_list = []
    for subFilesList in filesListSplitted:
        p = Process(target=preprocess_sample_list, args=(subFilesList, face_detector, landmark_detector, mean_face_landmarks, withaudio, defaultcrop))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()
