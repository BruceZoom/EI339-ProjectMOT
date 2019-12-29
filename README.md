# EI339-ProjectMOT

This is the repository for our work on the final project of EI339: Artificial Intelligence in Fall 2019.

## Structure

- `code`: The directory for all the source files
- `report.pdf`: The project report
- `code-explained.pdf`: A grading-friendly document that describes implementation of our ideas and modifications to open-source code.

## Dependencies
Following libraries are required to run our code.
- numpy
- scikit-learn
- opencv-python
- tensorflow >=1.0, <2.0
- torch >=1.3
- matplotlib
- pycocotools
- tqdm
- Pillow

Note that Tensorflow 2 is incompatible with the code for feature descriptor extraction, which is required for the real-time tracker.

Run the following command to install dependencies.
```bash
pip install -r code/requirements.txt
```

## Additional Resources
There are other resources you need to download to run our code.
- Download the weights for DeepSORT in 'resource.tar' from [JBox](https://jbox.sjtu.edu.cn/l/VooidI) and untar it as `code/resources`
- Download the weights for YOLOv3 also from [JBox](https://jbox.sjtu.edu.cn/l/VooidI) and place it under `code/yolov3/weights`

## Usage

To evaluate our implementation of Deep SORT with extended state space, download MOT16 dataset and run `code/deep_sort_app.py` according to the instructions in `code/README.md`. Assuming that the current working directory is `code`, here is an example:

```bash
python3 deep_sort_app.py \
    --sequence_dir=MOT16/train/MOT16-02 \
    --detection_file=resources/detections/MOT16_POI_train/MOT16-02.npy \
    --min_confidence 0.3 --nn_budget 100 --display True
```

To test the real-time tracker, assuming that the current working directory is `code` and your camera is available, run
```bash
python3 real_time_app.py
```
A window will display the tracking result of the camera input for 30 seconds.
