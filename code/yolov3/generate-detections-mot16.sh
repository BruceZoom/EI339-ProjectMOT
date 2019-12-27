#!/bin/bash

for seq_dir in ../MOT16/train/*; do
  python detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights \
    --source "$seq_dir/img1" --output "$seq_dir/det/det-yolov3.txt"
done
