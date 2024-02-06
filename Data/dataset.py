from PIL.Image import Image
import matplotlib
import glob
import torch
import tqdm
import pickle
import SimpleITK as sitk
import os
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
import numpy as np
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import lmdb
import time
import torch.nn.functional as F
import torchvision.transforms as tvt
import Data.transforms as TRANS
from multiprocessing import Pool, Manager
import multiprocessing
from typing import Callable, List, Optional, Tuple
from Lib import read_pickle, read_json, imap_tqdm
from functools import partial
from torchsampler import ImbalancedDatasetSampler
import copy


# TODO make a list in json format of your own data names
class mTICI_Dual_LMDB(data.Dataset):
    def __init__(
        self,
        json_file_dir="/ai/mnt/code/DSFNet_MTICI/Data/ReNamedAll.json",
        visual_size=512,
        fast_time_size=32,
        state="train",
        fuse01=True,
        use_trans=True,
        crop=(0.2, 0.2, 0.2, 0.2),
        binary=False,
        **kwargs
    ):
        if json_file_dir is None:
            json_file_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "ReNamedAll.json",
            )
        all_samples = read_json(json_file_dir)
        self.env = lmdb.open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "Relabeled_V%03d/" % visual_size,
            ),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.binary = binary
        self.fuse01 = fuse01
        if self.binary:
            self.fuse01 = False
        self.visual_size = visual_size
        self.state = state
        self.fast_time_size = fast_time_size
        self.crop = crop
        self.labels_list = None
        self.use_trans = use_trans

        self.samples = all_samples["CORONAL_VIEW"]["%s" % self.state]

        self.train_trans = TRANS.Compose(
            [
                TRANS.RandomErode(k=3, high=192, low=16, p=0.15),
                TRANS.RandomDilate(k=3, high=192, low=16, p=0.15),
                TRANS.TioClamp(clamps=(16, 192), p=0),
                TRANS.TioRandomFlip(p=0.25),
                TRANS.TioRandomAnisotropy(p=0.25),
                TRANS.TioRandomMotion(p=0.25),
                TRANS.TioRandomGhosting(p=0.25),
                TRANS.TioRandomSpike(p=0.25),
                TRANS.TioRandomBiasField(p=0.25),
                TRANS.TioRandomBlur(p=0.25),
                TRANS.TioRandomNoise(p=0.25),
                TRANS.TioRandomGamma(p=0.25),
                # TRANS.Extract_Temporal(
                #     raw_time_size=self.raw_time_size,
                #     target_time_size=self.target_time_size,
                # ),
                TRANS.RandomRotation(degrees=(-30, 30), fill=192, p=0.25),
                # TRANS.GaussianBlur(kernel_size=(21, 21)),
                TRANS.Crop(crop=self.crop),
                TRANS.Resize(
                    t=self.fast_time_size,
                    visual=(self.visual_size, self.visual_size),
                ),
                # TRANS.Fix_Normalize(),
                TRANS.TioZNormalization(p=1, div255=True),
            ]
        )
        self.val_trans = TRANS.TioZNormalization(p=1, div255=True)
        # if self.state == "train":
        #     self.trans = TRANS.Compose(
        #         [
        #             TRANS.clip_image(p=1),
        #             TRANS.RandomErase(p=1),
        #             TRANS.RandomRotation(degrees=(-30, 30), fill=192, p=0),
        #             TRANS.GaussianBlur(kernel_size=(21, 21)),
        #             TRANS.Crop(crop=self.crop),
        #             TRANS.Resize(
        #                 t=self.fast_time_size,
        #                 visual=(self.visual_size, self.visual_size),
        #             ),
        #             TRANS.Fix_Normalize(),
        #         ]
        #     )

        # else:
        #     self.trans = TRANS.Fix_Normalize()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.name = self.samples[index]
        self.label = self.__map_label(self.name)
        cor_view, sag_view = self.__load_arrays(
            self.name, fast_depth=self.fast_time_size, visual=self.visual_size
        )
        raw_cor_view = copy.deepcopy(cor_view)
        raw_sag_view = copy.deepcopy(sag_view)
        if self.use_trans and self.state == "train":
            cor_view = self.train_trans(cor_view).contiguous()
            sag_view = self.train_trans(sag_view).contiguous()
        else:
            cor_view = self.val_trans(cor_view).contiguous()
            sag_view = self.val_trans(sag_view).contiguous()

        return dict(
            cor_view=cor_view,
            sag_view=sag_view,
            raw_cor_view=raw_cor_view,
            raw_sag_view=raw_sag_view,
            label=self.label,
            name=self.name,
        )

    def get_weighted_count(self):
        labels_list = [0 for i in range(10)]
        for sample in self.samples:
            labels_list[self.__map_label(sample).item()] += 1
        labels_list = [i for i in labels_list if i]
        weighted_list = [
            len(self.samples) / labels_list[self.__map_label(i).item()]
            for i in self.samples
        ]
        self.labels_list = labels_list
        return weighted_list

    # TODO weather to read form dcm files or from lmdb
    def __load_arrays(self, file_dir, fast_depth, visual):
        with self.env.begin(write=False) as txn:
            cor_view = pickle.loads(
                txn.get(
                    (
                        "T%02d#V%03d#" % (fast_depth, visual)
                        + file_dir.split("/")[-1].replace(".pkl", ".dcm")
                    ).encode()
                )
            )
            sag_view = pickle.loads(
                txn.get(
                    (
                        "T%02d#V%03d#" % (fast_depth, visual)
                        + file_dir.split("/")[-1]
                        .replace(".pkl", ".dcm")
                        .replace("_C", "_S")
                    ).encode()
                )
            )
        return cor_view, sag_view

    # TODO weather to change the class nums
    def __map_label(self, file_dir):
        label_str = file_dir.split("/")[-1].split("_")[1]
        if self.binary:
            if label_str in ["T0", "T1", "T2a", "T2A"]:
                return torch.tensor(0).unsqueeze(0)
            elif label_str in ["T3", "T2B", "T2b"]:
                return torch.tensor(1).unsqueeze(0)
            else:
                assert False, "Label {} not supported".format(file_dir)
        else:
            if label_str == "T0":
                return torch.tensor(0).unsqueeze(0)
            elif label_str == "T1":
                return (
                    torch.tensor(1).unsqueeze(0)
                    if not self.fuse01
                    else torch.tensor(0).unsqueeze(0)
                )
            elif label_str == "T2a" or label_str == "T2A":
                return (
                    torch.tensor(2).unsqueeze(0)
                    if not self.fuse01
                    else torch.tensor(1).unsqueeze(0)
                )
            elif label_str == "T2b" or label_str == "T2B":
                return (
                    torch.tensor(3).unsqueeze(0)
                    if not self.fuse01
                    else torch.tensor(2).unsqueeze(0)
                )
            elif label_str == "T3":
                return (
                    torch.tensor(4).unsqueeze(0)
                    if not self.fuse01
                    else torch.tensor(3).unsqueeze(0)
                )
            else:
                assert False, "Label {} not supported".format(file_dir)


class mTICI_Single_LMDB(data.Dataset):
    def __init__(
        self,
        data_dir="/ai/mnt/code/DSFNet_MTICI/Data/",
        json_file_dir="/ai/mnt/code/DSFNet_MTICI/Data/train0.8_val_split.json",
        visual_size=(512, 512),
        time_stride=4,
        fast_time_size=32,
        slow_time_size=8,
        state="train",
        stop_flow_thresh=0.5,
        start_flow_thresh=0.5,
        flow_skip_frame=12,
        clip_flow=True,
        fuse01=False,
        over_sample=False,
        crop=(0.2, 0.2, 0.2, 0.2),
        cut_min_frame=8,
        pre_load=False,
        **kwargs
    ):
        all_samples = read_json(json_file_dir)
        self.env = lmdb.open(
            data_dir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.visual_size = visual_size
        self.state = state
        self.cut_min_frame = cut_min_frame
        self.clip_flow = clip_flow
        self.time_stride = time_stride
        self.fast_time_size = fast_time_size
        self.slow_time_size = slow_time_size
        self.stop_flow_thresh = stop_flow_thresh
        self.start_flow_thresh = start_flow_thresh
        self.flow_skip_frame = flow_skip_frame
        self.crop = crop
        self.fuse01 = fuse01
        if self.state == "train":
            self.samples = all_samples["single_view_train"]
        else:
            self.samples = all_samples["single_view_val"]

        if self.state == "train":
            # if over_sample:
            #     self.samples = TRANS.get_ovesample_images(
            #         self.samples,
            #         self.fuse01,
            #     )

            self.trans = TRANS.Compose(
                [
                    TRANS.clip_image(p=1),
                    TRANS.RandomErase(p=1),
                    TRANS.RandomRotation(degrees=(-30, 30), fill=192, p=0),
                    TRANS.GaussianBlur(kernel_size=(21, 21)),
                    TRANS.Crop(crop=self.crop),
                    TRANS.Fix_Normalize(),
                ]
            )

        else:
            self.trans = TRANS.Fix_Normalize()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.name = self.samples[index]
        self.label = self.__map_label(self.name)
        fast_view, slow_view = self.__load_arrays(
            self.name,
            fast_depth=self.fast_time_size,
            slow_depth=self.slow_time_size,
        )

        fast_view = self.trans(fast_view).contiguous()
        slow_view = self.trans(slow_view).contiguous()

        return dict(
            fast_view=fast_view,
            slow_view=slow_view,
            label=self.label,
            name=self.name,
        )

    def get_weighted_count(self):
        labels_list = [0 for i in range(10)]
        for sample in self.samples:
            labels_list[self.__map_label(sample).item()] += 1
        labels_list = [i for i in labels_list if i]
        weighted_list = [
            len(self.samples) / labels_list[self.__map_label(i).item()]
            for i in self.samples
        ]
        return weighted_list

    def __load_arrays(self, file_dir, fast_depth, slow_depth):
        with self.env.begin(write=False) as txn:
            fast_view = pickle.loads(
                txn.get(
                    ("%02d#" % fast_depth + file_dir.replace(".pkl", ".dcm")).encode()
                )
            )
            slow_view = pickle.loads(
                txn.get(
                    ("%02d#" % slow_depth + file_dir.replace(".pkl", ".dcm")).encode()
                )
            )
        return fast_view, slow_view

    def __map_label(self, file_dir):
        label_str = file_dir.split("_")[3]
        if label_str == "T0":
            return torch.tensor(0).unsqueeze(0)
        elif label_str == "T1":
            return (
                torch.tensor(1).unsqueeze(0)
                if not self.fuse01
                else torch.tensor(0).unsqueeze(0)
            )
        elif label_str == "T2a":
            return (
                torch.tensor(2).unsqueeze(0)
                if not self.fuse01
                else torch.tensor(1).unsqueeze(0)
            )
        elif label_str == "T2b":
            return (
                torch.tensor(3).unsqueeze(0)
                if not self.fuse01
                else torch.tensor(2).unsqueeze(0)
            )
        elif label_str == "T3":
            return (
                torch.tensor(4).unsqueeze(0)
                if not self.fuse01
                else torch.tensor(3).unsqueeze(0)
            )
        else:
            assert False, "Label {} not supported".format(label_str)

    def __clip_flow_img(
        self,
        content,
        stop_flow_thresh=0.5,
        start_flow_thresh=0.5,
        flow_skip_frame=12,
        cut_min_frame=8,
    ):
        view = content["view"]
        start_mean_acc = content["start_mean_acc"]
        stop_mean_acc = content["stop_mean_acc"]
        raw_shape = view.shape[0]
        if (
            raw_shape > flow_skip_frame
            and start_flow_thresh > 0
            and stop_flow_thresh > 0
        ):
            start_index = 1000
            stop_index = -1
            for i in range(len(start_mean_acc)):
                if start_mean_acc[i] > start_flow_thresh:
                    start_index = min(start_index, i)
            for i in range(len(stop_mean_acc)):
                if stop_mean_acc[i] > stop_flow_thresh:
                    stop_index = max(stop_index, i)
            if stop_index == -1:
                stop_index = raw_shape - 1
            if start_index == 1000:
                start_index = 0
            if stop_index - start_index + 1 > cut_min_frame:
                view = view[start_index:stop_index, ...]
        return view

    def __resize_array(
        self, view_lists, visual_size, fast_time_size, slow_time_size, time_stride
    ):
        resized_views = []

        for view in view_lists:
            view = view.unsqueeze(0)
            fast_view = F.interpolate(
                view,
                (fast_time_size, visual_size[0], visual_size[1]),
                mode="trilinear",
                align_corners=True,
            ).contiguous()[0]
            slow_view = F.interpolate(
                view,
                (slow_time_size, visual_size[0], visual_size[1]),
                mode="trilinear",
                align_corners=True,
            ).contiguous()[0]
            raw_view = F.interpolate(
                view,
                (view.shape[2], visual_size[0], visual_size[1]),
                mode="trilinear",
                align_corners=True,
            ).contiguous()[0]
            resized_views.append(fast_view)
            resized_views.append(slow_view)
            resized_views.append(raw_view)
        return tuple(resized_views)


if __name__ == "__main__":
    visuals = [256]
    temporals = [8]

    tici = mTICI_Dual_LMDB(
        state="train",
        fast_time_size=8,
        visual_size=256,
        fuse01=False,
        json_file_dir=None,
        binary=False,
    )
    print(tici.get_weighted_count())
    print("TESTING WITH T%02d and V%03d" % (8, 256))
    start = time.time()
    labels_list = [0 for i in range(10)]
    for index, datas in tqdm.tqdm(enumerate(tici)):
        labels_list[datas["label"].item()] += 1
    fps = len(tici) / (time.time() - start)
    print(fps)
    print(labels_list)
