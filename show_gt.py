#!/usr/bin/env python


import cv2
import numpy as np
import os

def load_mot(groundtruth):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).
    Args:
        detections
    Returns:
        list: list containing the detections for each frame.
    """

    data = []
    if type(groundtruth) is str:
        raw = np.genfromtxt(groundtruth, delimiter=',', dtype=np.float32)
    else:
        # assume it is an array
        assert isinstance(groundtruth, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = groundtruth.astype(np.float32)

    end_frame = int(np.max(raw[:, 0]))
    for i in range(1, end_frame+1):
        idx = raw[:, 0] == i
        # print(i,idx)
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = raw[idx, 6]
        gts = []
        for bb, s in zip(bbox, scores):
            gts.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s})
        data.append(gts)

    return data

def main():
    dataset_index = [2, 4, 5, 9, 10, 11, 13]
    dataset_detection_type = {'-DPM', '-FRCNN', '-SDP'}

    dataset_image_folder_format = os.path.join('./dataset/MOT17',
                                               'train' + '/MOT' + str(17) + '-{:02}{}/img1')
    groundtruth_file_name_format = os.path.join('./dataset/MOT17',
                                              'train' + '/MOT' + str(17) + '-{:02}{}/gt/gt.txt')

    save_folder = './seagate/logs/gt'

    save_video_name_format = os.path.join(save_folder, 'MOT' + str(17) + '-{:02}{}.avi')

    f = lambda format_str: [format_str.format(index, type) for type in dataset_detection_type for index in
                            dataset_index]

    for image_folder, groundtruth_file_name, save_video_name in zip(f(dataset_image_folder_format),
                                                                                   f(groundtruth_file_name_format),
                                                                                   f(save_video_name_format)):

        first_run = True
        image_format = os.path.join(image_folder, '{0:06d}.jpg')
        gts = load_mot(groundtruth_file_name)

        for frame_num, groundtruths_frame in enumerate(gts, start=1):
            # 读取视频
            img = cv2.imread(image_format.format(frame_num))
            h, w, _ = img.shape
            if first_run:
                vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))
                first_run = False

            for a in range(len(groundtruths_frame)):
                bbox = groundtruths_frame[a]["bbox"]
                print(frame_num, "bbox:x1, y1, x2, y2", bbox)
                # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                # 将解析处理的矩形框，绘制在视频上，实时显示
                cv2.rectangle(img, bbox[:2], bbox[2:], (255, 0, 0), 2)
            cv2.imshow("frame", img)
            vw.write(img)

            # 键盘控制视频播放  waitKey(x)控制视频显示速度
            key = cv2.waitKey(100) & 0xFF

            if key == ord(' '):
                cv2.waitKey(0)
            if key == ord('q'):
                break


if __name__ == '__main__':
    main()
