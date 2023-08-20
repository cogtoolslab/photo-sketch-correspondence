import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class PhotoSketchDataset(Dataset):
    """
    Dataset that loads photo-sketch pairs during training, evaluation, and testing.
    Since each photo corresponds to multiple sketches in the dataset, it loops through
    all photos and sample a sketch from each photo to construct the photo-sketch pair.
    """

    def __init__(self, csv_path, data_path, mode, mix_within_class=False, num_photo_per_class=90):
        """
        :param csv_path: path to the train/test pair csv file
        :param data_path: root path of the dataset
        :param mode: 1) train: (used for training)
                               returns photo, sketch, index of class, and a flip (or not) parameter
                     2) eval: (only used for visualization during training)
                              returns photo, sketch, index of class, and 0 (no flipping)
                     3) test: (used for PCK error metric calculation)
                              returns photo, sketch, keypoints in photo, keypoints in sketch
        :param mix_within_class: False: it will sample a photo and a sketch from the same photo-sketch pair
                                 True: it will sample a photo and a sketch from the same category
        :param num_photo_per_class: number of photos in each class. Used only if mix_within_class is True.
        """

        self.csv = pd.read_csv(csv_path)
        self.data_path = data_path
        self.mode = mode
        self.mix_within_class = mix_within_class
        self.unique_photo_ids = self.csv["photo"].unique()

        self.trans = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_photo_per_class = num_photo_per_class

    def __len__(self):
        if self.mode != "test":
            return len(self.unique_photo_ids)
        else:
            return len(self.csv)

    def _load_train_pair(self, idx):
        # get the photo_id of the photo
        photo_id = self.unique_photo_ids[idx]

        # sample a sketch from sketches corresponding to the photo
        row = self.csv[self.csv["photo"] == photo_id].sample().iloc[0]

        # load photo and sketch
        photo = np.array(Image.open(os.path.join(self.data_path, row["photo"])))
        sketch = np.array(Image.open(os.path.join(self.data_path, row["sketch"])))
        class_idx = row["class"]
        flip = row["flip"]
        return photo, sketch, class_idx, flip

    def _load_eval_pair(self, idx):
        photo_id = self.unique_photo_ids[idx]
        row = self.csv[self.csv["photo"] == photo_id].sample().iloc[0]
        photo = np.array(Image.open(os.path.join(self.data_path, row["photo"])))
        sketch = np.array(Image.open(os.path.join(self.data_path, row["sketch"])))
        class_idx = row["class"]
        return photo, sketch, class_idx, 0

    def _load_test_pair(self, idx):
        row = self.csv.iloc[idx]
        photo = np.array(Image.open(os.path.join(self.data_path, row["photo"])))
        sketch = np.array(Image.open(os.path.join(self.data_path, row["sketch"])))

        # interpret the keypoint XY from the string in csv
        y1 = [float(v) for v in row["XA"].split(";")]
        x1 = [float(v) for v in row["YA"].split(";")]
        y2 = [float(v) for v in row["XB"].split(";")]
        x2 = [float(v) for v in row["YB"].split(";")]

        kp1 = np.stack([x1, y1], axis=1)
        kp2 = np.stack([x2, y2], axis=1)
        return photo, sketch, kp1, kp2

    def __getitem__(self, idx):
        if self.mode == "train":
            return self._get_train_item(idx)
        elif self.mode == "eval":
            return self._get_eval_item(idx)
        elif self.mode == "test":
            return self._get_test_item(idx)
        else:
            raise NotImplementedError

    def _get_train_item(self, idx):

        if self.mix_within_class:
            photo, sketch, class_idx, flip = self._load_train_pair(idx)
            new_idx = (class_idx - 1) * self.num_photo_per_class + np.random.randint(self.num_photo_per_class)
            _, sketch, _, _ = self._load_train_pair(new_idx)
        else:
            photo, sketch, class_idx, flip = self._load_train_pair(idx)

        photo = self.trans(photo)
        sketch = self.trans(sketch)

        if flip == 1:
            photo = TF.hflip(photo)
            sketch = TF.hflip(sketch)

        return photo, sketch

    def _get_eval_item(self, idx):

        photo, sketch, class_idx, _ = self._load_eval_pair(idx)

        photo = self.norm(self.trans(photo))
        sketch = self.norm(self.trans(sketch))

        return photo, sketch

    def _get_test_item(self, idx):

        photo, sketch, kp1, kp2 = self._load_test_pair(idx)

        photo = self.norm(self.trans(photo))
        sketch = self.norm(self.trans(sketch))

        return photo, sketch, kp1, kp2
