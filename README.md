# SegNet- UBC Implementation
This is modified implementation of Segnet inspired by [segmentation network proposed by Kendall et al.](http://mi.eng.cam.ac.uk/projects/segnet/) on UBC Human Pose dataset and Berkely MHAD.

Implementations of U-Net, Pointnets(++) are next in line for the whole task of pose estimation.

The I/O pipelines for both the datasets are ready.

In the `config.py` file, the `dataset_name` needs to match the data directories you create in your `input` folder. You can use `UBC_easy` and `segnet-32`.

## Train and test
Generate your TFRecords using `tfrecorder.py`. Make sure the dataset is downloaded in and make it compatible with the directories mentioned in the `tfrecorder.py` file.

Once you have your TFRecords, train SegNet with `python train.py`. Analogously, test it with `python test.py`.

Also, the regression pipeline for both of the above datasets are written in MATLAB, and will be up in a separate repo shortly. In case of any questions, drop me an email at `ashar.ali@gatech.edu`