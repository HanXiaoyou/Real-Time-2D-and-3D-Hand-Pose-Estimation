"""
RHD testset
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
import pickle

import torch
import torch.utils.data

from hand_shape_pose.util.image_util import crop_pad_im_from_bounding_rect

resize_dim = [256, 256]

class RHD_test(torch.utils.data.Dataset):
    def __init__(self, root, image_dir):
        self.image_paths = []
        self.data_path = root

        self.cam_params, self.pose_scales, self.pose_gts = self.load_db_annotation(root, "evaluation")

        num_test = int(self.cam_params.shape[0]/2)
        for image_id in range(num_test):
            self.image_paths.append(osp.join(image_dir, "%05d.png" % (image_id)))
            self.image_paths.append(osp.join(image_dir, "%05d.png" % (image_id)))
        # self.pose_gts = self.load_db_annotation(self, root, set_name="")
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


    def compute_hand_scale(self, pose_xyz):
        # from tip to palm : left index : 7-8 right index : 28-29
        ref_bone_joint_1_id = pose_xyz[(7, 28), :]
        ref_bone_joint_2_id = pose_xyz[(8, 29), :]

        pose_scale_vec = ref_bone_joint_1_id - ref_bone_joint_2_id  # N x 3
        pose_scale_vec = torch.Tensor(pose_scale_vec).float().cuda()
        pose_scale = torch.norm(pose_scale_vec, dim=1)  # N
        pose_scale = pose_scale.cpu().numpy()
        return pose_scale

    def load_db_annotation(self, base_path, set_name=None):
        if set_name is None:
            # only training set annotations are released so this is a valid default choice
            set_name = 'training'

        print('Loading RHD dataset index ... and the path is ', base_path)
        #t = time.time()

        # load if exist
        # assumed paths to data containers
        with open(os.path.join(base_path, set_name, 'anno_%s.pickle' % set_name), 'rb') as fi:
            anno_all = pickle.load(fi)

        K_list = []
        scale_list = []  # mean: 0.0399
        xyz_list = []
        for sample_id, anno in anno_all.items():
            K_list.append(anno['K'])
            K_list.append(anno['K'])
            xyz_list.append(anno['xyz'][0:21])
            xyz_list.append(anno['xyz'][21:42])
            scale_double = self.compute_hand_scale(anno['xyz'])  # left and right hand scales
            scale_list.append(scale_double[0])
            scale_list.append(scale_double[1])

        # should have all the same length
        assert len(K_list) == len(scale_list), 'Size mismatch.'

        print('Loading of %d samples done' % (len(K_list)))
        #print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
        return torch.Tensor(K_list), torch.Tensor(scale_list), torch.Tensor(xyz_list)

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

        return avg_est_error.item()