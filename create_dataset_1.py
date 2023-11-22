"""
This script to create .csv videos frames action annotation file.

- It will play a video frame by frame control the flow by [a] and [d]
 to play previos or next frame.
- Open the annot_file (.csv) and label each frame of video with number
 of action class.
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
import torch
from fn import vis_frame_fast

# DETECTION MODEL.
detector = TinyYOLOv3_onecls()
# POSE MODEL.
inp_h = 320
inp_w = 256
pose_estimator = SPPE_FastPose('resnet50', inp_h, inp_w)

class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
               'Stand up', 'Sit down', 'Fall Down']  # label.

video_folder = 'Data/falldata/Home/Videos'
annot_file = 'Data/Home_new.csv'

index_video_to_play = 0  # Choose video to play.


def create_csv(folder):
    list_file = sorted(os.listdir(folder))
    cols = ['video', 'frame', 'label']
    df = pd.DataFrame(columns=cols)
    for fil in list_file:
        cap = cv2.VideoCapture(os.path.join(folder, fil))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = np.array([fil] * frames_count)
        frame = np.arange(1, frames_count + 1)
        label = np.array([0] * frames_count)
        rows = np.stack([video, frame, label], axis=1)
        df = df._append(pd.DataFrame(rows, columns=cols),
                       ignore_index=True)
        cap.release()
    df.to_csv(annot_file, index=False)


# if not os.path.exists(annot_file):
#     create_csv(video_folder)
df = pd.DataFrame(columns=['video', 'frame', 'label'])

# annot = pd.read_csv(annot_file)
# video_list = annot.iloc[:, 0].unique()
# video_file = os.path.join(video_folder, video_list[index_video_to_play])
# print(os.path.basename(video_file))
list_file = sorted(os.listdir(video_folder))
i = 0
for fil in list_file:

    # annot = annot[annot['video'] == video_list[index_video_to_play]].reset_index(drop=True)
    # frames_idx = annot.iloc[:, 1].tolist()

    # cap = cv2.VideoCapture(video_file)
    # frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # assert frames_count == len(frames_idx), 'frame count not equal! {} and {}'.format(
    #     len(frames_idx), frames_count
    # )
    video_file = os.path.join(video_folder, fil)
    print(video_file)
    cap = cv2.VideoCapture(video_file)

    cls_idx = 0
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 1.5,
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 1.5)

        if ret:
            # cls_name = class_names[int(annot.iloc[i, -1]) - 1]
            frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
            # frame = cv2.putText(frame, 'Frame: {} Pose: {}'.format(i+1, cls_name),
            #                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = cv2.putText(frame, 'Frame: {}'.format(frame_idx+1),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # !-- Add
            detect_result = detector.detect(frame)
            if detect_result != None:
                bb = detect_result[0, :4].numpy().astype(int)
                bb[:2] = np.maximum(0, bb[:2] - 5)
                bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]
                if bb.any() != 0:
                    result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                                    torch.tensor([[1.0]]))
                    if len(result) > 0:
                        scr = result[0]['kp_score'].mean()
                        frame = vis_frame_fast(frame, result)
                        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, cls_idx, scr),
                                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                i += 1
                frame_idx += 1
                cls_idx = 99
                df.loc[i] = [fil, frame_idx, cls_idx]
                print('Frame: {} Pose: None'.format(frame_idx))
                continue

            cv2.imshow('frame', frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                i += 1
                frame_idx += 1
                continue
            elif key == ord('a'):
                i -= 1
                frame_idx -= 1
                continue
            # !-- Add
            elif key == ord('0'): # Standing
                cls_idx = 0
            elif key == ord('1'): # Walking
                cls_idx = 1
            elif key == ord('2'): # Sitting
                cls_idx = 2
            elif key == ord('3'): # Lying Down
                cls_idx = 3
            elif key == ord('4'): # Stand up
                cls_idx = 4
            elif key == ord('5'): # Sit down
                cls_idx = 5
            elif key == ord('6'): # Fall down
                cls_idx = 6
            elif key == ord('x'): # None
                i += 1
                frame_idx += 1
                cls_idx = 99
                df.loc[i] = [fil, frame_idx, cls_idx]
                print('Frame: {} Pose: None'.format(frame_idx))
                continue

            i += 1
            frame_idx += 1
            cls_name = class_names[cls_idx]
            df.loc[i] = [fil, frame_idx, cls_idx]
            print('Frame: {} Pose: {}'.format(frame_idx, cls_name))
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# !-- Add
if os.path.exists(annot_file):
    df.to_csv(annot_file, mode='a', header=False, index=False)
else:
    df.to_csv(annot_file, mode='w', index=False)