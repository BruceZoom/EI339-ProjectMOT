import os
import cv2
import sys


def video2frame(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(1, 1001):
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(output_dir, "{:06}.jpg".format(i)), frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 3: print("Too few arguments!"); exit(-1)
    vpath = sys.argv[1]
    odir = sys.argv[2]
    video2frame(vpath, odir)
