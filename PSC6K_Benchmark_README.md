# Photo-Sketch Correspondence Benchmarking (*PSC6K*)

## Overview
We developed ***PSC6k***, a photo-sketch correspondence dataset consisting of **150,000 pairs of keypoint annotations** 
for **6250 photo-sketch pairs** spanning **125 object categories**. It is built by annotating the keypoint 
correspondences for the [Sketchy dataset](sketchy.eye.gatech.edu), which contains 75,471 human sketches produced from 
12,500 unique photographs.

## Download
**PSC6K Benchmark** (keypoint annotations): [Amazon S3](https://photo-sketch-correspondence-6k.s3.amazonaws.com/PSC6K.zip)

Since the image data are not included in our release, 
please use links below to download all photo-sketch images.

**Sketchy Dataset** (photo-sketch images): [Official Page](sketchy.eye.gatech.edu) | 
[Google Drive](https://drive.google.com/file/d/1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck/view)

===========================

Our PSC6K benchmark release includes:
* The Train/Test Splits and Keypoint Annotations
* Two Example PyTorch Dataset Implementations.
* An Example Jupyter Notebook to Explore the Dataset

## Train/Test Splits and Keypoint Annotations

We present the keypoint annotation data in the same format as PF-Pascal, which is widely used in existing 
correspondence estimation methods.

### Photo/Sketch format
`train_pairs_ps.csv` records the train split of the Sketchy Dataset. Each row represents a photo-sketch pair. 
The file has 4 columns:
* **photo**: the relative path to the photo in the Sketchy dataset
* **sketch**: the relative path to the sketch in the Sketchy dataset
* **class**: the index of class for the photo-sketch pair
* **flip**: whether a horizontal flip augmentation should be applied to this photo-sketch pair

Here, each photo-sketch pair always corresponds to 2 rows in the file, one with `flip=0` and the other with `flip=1`.

`test_pairs_ps.csv` records the test split of the Sketchy Dataset and all the keypoint annotations. Each row represents 
a photo-sketch pair. The file has 7 columns:
* **photo**: the relative path to the photo in the Sketchy dataset
* **sketch**: the relative path to the sketch in the Sketchy dataset
* **class**: the index of class for the photo-sketch pair
* **XA**: the list of X coordinates of keypoints in the photo
* **YA**: the list of Y coordinates of keypoints in the photo
* **XB**: the list of X coordinates of keypoints in the sketch
* **YB**: the list of Y coordinates of keypoints in the sketch

There are 8 keypoints for each photo-sketch pair. All coordinates in the list are seperated by semicolons.

### Source/Target Format
Since most correspondence estimation methods does not read images in the form of photo-sketch pair, 
but in the form of source-target pair, we also provide a source-target version for easier evaluation on these methods.

`train_pairs_st.csv` records the train split of the Sketchy Dataset. Each row represents a photo-sketch pair. 
The file has 4 columns:
* **source_image**: the relative path to the source image in the Sketchy dataset
* **target_image**: the relative path to the target image in the Sketchy dataset
* **class**: the index of class for the photo-sketch pair
* **flip**: whether a horizontal flip augmentation should be applied to this photo-sketch pair

Here, Each photo-sketch pair always corresponds to 4 rows in the file, which covers the 2 `flip` conditions and 
the 2 `source->target` directions (`photo->sketch` and `sketch->photo`).


`test_pairs_st.csv` records the test split of the Sketchy Dataset and all the keypoint annotations. 
Each row represents a photo-sketch pair. The file has 7 columns:
* **source_image**: the relative path to the source image in the Sketchy dataset
* **target_image**: the relative path to the target image in the Sketchy dataset
* **class**: the index of class for the photo-sketch pair
* **XA**: the list of X coordinates of keypoints in the source image
* **YA**: the list of Y coordinates of keypoints in the source image
* **XB**: the list of X coordinates of keypoints in the target image
* **YB**: the list of Y coordinates of keypoints in the target image

There are 8 keypoints for each photo-sketch pair. All coordinates in the list are seperated by semicolons. 
Each photo-sketch pair always corresponds to 2 rows in the file, one being `photo->sketch` 
and the other being `sketch->photo`.


## Example PyTorch Dataset Implementations
We provide two examples of PyTorch dataset implementations:
* `alignment_dataset.py` follows the testing dataset implementation in existing correspondence estimation methods. 
It is modified from the PF-Pascal dataloader in [WeakAlign](https://github.com/ignacio-rocco/weakalign), 
and can be easily plugged into existing methods with minor changes for evaluation. 
It requires NumPy, Pandas, Skimage, and PyTorch to run.

* `photo_sketch_dataset.py` is the dataset structure that we used during training. 
It can load photo/sketch samples from the train split, and load photo/sketch/annotation samples from the test split. 
It requires NumPy, PIL, Pandas, and PyTorch to run.


## Example Jupyter Notebook
Lastly, we provide a Jupyter Notebook `dataset_demo.ipynb` that helps you to explore the *PSC6K* dataset. 
It exhibits a basic pipeline for folder path setup, dataset loading, and keypoint visualization.
