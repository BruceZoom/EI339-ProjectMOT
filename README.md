# EI339-ProjectMOT

This is the repository for our work on the final project of EI339: Artificial Intelligence in Fall 2019.

## Structure

- `code`: The directory for all the source files
- `report.pdf`: The project report
- `code-explained.pdf`: A grading-friendly document that describes implementation of our ideas and modifications to open-source code.

## Usage

Make sure that dependencies specified in `requirements.txt` are installed. Note that Tensorflow 2 is incompatible with the code for feature descriptor extraction, which is required for the real-time tracker.

To evaluate our implementation of Deep SORT with extended state space, download MOT16 dataset and run `code/deep_sort_app.py` according to the instructions in `code/README.md`. Assuming that the current working directory is `code`, here is an example:

```bash
python3 deep_sort_app.py \
    --sequence_dir=MOT16/train/MOT16-02 \
    --detection_file=resources/detections/MOT16_POI_train/MOT16-02.npy \
    --min_confidence 0.3 --nn_budget 100 --display True
```

To test the real-time tracker, download the weights for YOLOv3 from [JBox](https://jbox.sjtu.edu.cn/l/b1kNny) and place it under `code/yolov3/weights`. Assuming that the current working directory is `code`, run

```bash
python3 real_time_app.py
```
