"""
FreiHAND testset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.io as sio
import os.path as osp
import logging
import cv2
import numpy as np
import numpy.linalg as LA
import math
import json
import os

import torch
import torch.utils.data

from hand_shape_pose.util.image_util import crop_pad_im_from_bounding_rect

resize_dim = [256, 256]

class FreiHANDTestset(torch.utils.data.Dataset):
    def __init__(self, root, image_dir):
        self.image_paths = []
        self.data_path = root

        self.cam_params, self.pose_scales, self.pose_gts = self.load_db_annotation(root, "evaluation")

        for image_id in range(self.cam_params.shape[0]):
            self.image_paths.append(osp.join(image_dir, "%08d.jpg" % (image_id)))

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        resized_img = cv2.resize(img, (resize_dim[1], resize_dim[0]))
        resized_img = torch.from_numpy(resized_img)  # 256 x 256 x 3

        return resized_img, self.cam_params[index], self.pose_scales[index], index

    def __len__(self):
        return len(self.image_paths)

    def _assert_exist(self, p):
        msg = 'File does not exists: %s' % p
        assert os.path.exists(p), msg

    def json_load(self, p):
        self._assert_exist(p)
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d

    def projectPoints(self, xyz, K):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return torch.Tensor(uv[:, :2]/uv[:, -1:])

    def db_size(self, set_name):
        """ Hardcoded size of the datasets. """
        if set_name == 'training':
            return 32560  # number of unique samples (they exists in multiple 'versions')
        elif set_name == 'evaluation':
            return 200
        else:
            assert 0, 'Invalid choice.'

    def load_db_annotation(self, base_path, set_name=None):
        if set_name is None:
            # only training set annotations are released so this is a valid default choice
            set_name = 'training'

        print('Loading FreiHAND dataset index ...')
        #t = time.time()

        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

        # load if exist
        K_list = self.json_load(k_path)
        scale_list = self.json_load(scale_path)
        xyz_list = self.json_load(xyz_path)

        # should have all the same length
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
        assert len(K_list) == len(scale_list), 'Size mismatch.'

        print('Loading of %d samples done' % (len(K_list)))
        #print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
        # return torch.Tensor(K_list), torch.Tensor(scale_list)
        return torch.Tensor(K_list), torch.Tensor(scale_list), torch.Tensor(xyz_list)[:, :, :]

    def evaluate_pose(self, results_pose_cam_xyz, save_results=False, output_dir=""):
        avg_est_error = 0.0
        for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
            dist = est_pose_cam_xyz - self.pose_gts[image_id]  # K x 3
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_cam_xyz)

        if save_results:
            eval_results = {}
            image_ids = list(results_pose_cam_xyz.keys())
            image_ids.sort()
            eval_results["image_ids"] = np.array(image_ids)
            eval_results["gt_pose_xyz"] = [self.pose_gts[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["est_pose_xyz"] = [results_pose_cam_xyz[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["gt_pose_xyz"] = torch.cat(eval_results["gt_pose_xyz"], 0).numpy()
            eval_results["est_pose_xyz"] = torch.cat(eval_results["est_pose_xyz"], 0).numpy()
            sio.savemat(osp.join(output_dir, "pose_estimations.mat"), eval_results)

        return avg_est_error.item(), self.pose_gts

    def evaluate_3d_pck(self, results_pose_cam_xyz, thresholds, output_dir=""):
        pck_list = []
        for thr in thresholds:
            joint_threshold = np.ones(21)*thr
            num_joints_under_threshold = 0
            for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
                under_threshold = np.zeros(21)
                dist = est_pose_cam_xyz - self.pose_gts[image_id]  # K x 3
                joint_est_error = dist.pow(2).sum(-1).sqrt()
                joint_est_error = np.asarray(joint_est_error)
                under_threshold[joint_est_error < joint_threshold] = 1
                num_joints_under_threshold += under_threshold.sum()
            pck_list.append(num_joints_under_threshold/(len(results_pose_cam_xyz)*21))

        return pck_list
