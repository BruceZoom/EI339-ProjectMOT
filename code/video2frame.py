import os
import cv2
import sys
import numpy as np

def video2frame(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    idx = 0
    while(idx < 1000):
        print("Current frame: {}".format(idx))
        ret, frame = cap.read()
        if idx % 5 == 0:
            cv2.imwrite(os.path.join(output_dir, "{}.jpg".format(idx // 5)), frame)
        idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 3: print("Too few arguments!"); exit(-1)
    vpath = sys.argv[1]
    odir = sys.argv[2]
    video2frame(vpath, odir)
