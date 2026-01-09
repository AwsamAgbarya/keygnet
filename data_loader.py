import os 
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import platform
# import open3d as o3d


def rgbd_to_point_cloud(K, depth, rgb):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs, rgb[vs, us, 0], rgb[vs, us, 1], rgb[vs, us, 2]]).T
    return pts


class BOPDataset(Dataset):
    def __init__(self, root, set, min_visible_points=2000, points_count_net=1024) -> None:
        super().__init__()
        self.root = root
        self.set = set
        self.points_count_net = points_count_net
        self.cycle_path = os.path.join(root, set)
        self.split_path = os.path.join(self.root, "split")
        # standarization of ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float64)

        # generate splits
        if not os.path.exists(self.split_path):
            os.mkdir(self.split_path)
        if not os.path.isfile(os.path.join(self.split_path, set + '.txt')):
            print("Split txt not exists, generating ", self.set, " split file...")
            split_txt = open(os.path.join(self.split_path, set + '.txt'), 'w')
            for cycle in os.listdir(self.cycle_path):
                if os.path.isdir(os.path.join(self.cycle_path, cycle)):
                    for mask_name in os.listdir(os.path.join(self.cycle_path, cycle, 'mask_visib')):
                        mask = np.array(Image.open(os.path.join(self.cycle_path, cycle, 'mask_visib', mask_name)))
                        if np.count_nonzero(mask) >= min_visible_points:
                            split_txt.writelines(cycle + '/' + str(mask_name.split('.')[0]).zfill(6) + '\n')
            split_txt.close()
        with open(os.path.join(self.split_path, self.set + ".txt")) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]
        # generate normalization parameters
        self.cad_models_path = os.path.join(self.root, "models")
        cad_models_info_json = open(os.path.join(self.cad_models_path, 'models_info.json'), 'r')
        cad_models_info = json.load(cad_models_info_json)
        # x, y, z
        self.coor_dims = np.zeros(3)
        for i in cad_models_info:
            if self.coor_dims[0] < cad_models_info[i]['size_x'] / 2:
                self.coor_dims[0] = cad_models_info[i]['size_x'] / 2
            if self.coor_dims[1] < cad_models_info[i]['size_y'] / 2:
                self.coor_dims[1] = cad_models_info[i]['size_y'] / 2
            if self.coor_dims[2] < cad_models_info[i]['size_z'] / 2:
                self.coor_dims[2] = cad_models_info[i]['size_z'] / 2
        # dump normalization parameters
        np.savetxt(os.path.join(self.split_path, 'coor_dims.txt'), self.coor_dims)
        # print(self.coor_dims)

    def transform(self, rgb, depth, cam_k):
        # rgb -= self.mean
        # rgb /= self.std
        pts = rgbd_to_point_cloud(cam_k, depth, rgb)

        # pts: (N, 6) -> xyz + rgb
        xyz = pts[:, 0:3]
        colors = pts[:, 3:]

        # approximate radius outlier removal (replacement for Open3D remove_radius_outlier)
        # keep points that have enough neighbors within a radius
        if xyz.shape[0] > 0:
            from sklearn.neighbors import NearestNeighbors  # scikit-learn is already installed in your env
            radius = 20.0
            nb_points = 15
            nbrs = NearestNeighbors(radius=radius)
            nbrs.fit(xyz)
            neighborhoods = nbrs.radius_neighbors(xyz, return_distance=False)
            inlier_mask = np.array([len(n) >= nb_points for n in neighborhoods], dtype=bool)
            xyz = xyz[inlier_mask]
            colors = colors[inlier_mask]

        pts_tmp = xyz
        pts = np.zeros((pts_tmp.shape[0], 6))
        pts[:, 0:3] = pts_tmp
        pts[:, 3:] = colors
        if pts.shape[0] < self.points_count_net:
            pts = np.concatenate((pts, np.zeros((self.points_count_net - pts.shape[0], 6))), axis=0)
        else:
            idx = np.random.choice(np.arange(pts.shape[0]), self.points_count_net, replace=False)
            pts = pts[idx]
        for i in range(3):
            pts[:, i] -= np.mean(pts[:, i])
            pts[:, i] /= self.coor_dims[i]
        pts = torch.from_numpy(pts).float()
        return pts

    def __getitem__(self, index):
        id = self.ids[index]
        cycle, scene_objidx = id.split('/')
        scene, objidx = scene_objidx.split('_')
        if self.set == 'train':
            rgb = np.array(Image.open(os.path.join(self.cycle_path, cycle, 'rgb', scene + '.png')))
        else:
            rgb = np.array(Image.open(os.path.join(self.cycle_path, cycle, 'rgb', scene + '.png')))
        rgb = rgb.astype('float64')
        rgb /= 255.

        depth = np.array(Image.open(os.path.join(self.cycle_path, cycle, 'depth', scene + '.png')))
        mask_visb = np.array(Image.open(os.path.join(self.cycle_path, cycle, 'mask_visib', scene_objidx + '.png')))
        cam_k_json = json.load(open(os.path.join(self.cycle_path, cycle, 'scene_camera.json'), 'r'))
        cam_k = np.array(cam_k_json[str(int(scene))]['cam_K'])
        cam_k = cam_k.reshape(3, 3)
        depth_scale = cam_k_json[str(int(scene))]['depth_scale']
        depth = depth * depth_scale
        depth = np.where(mask_visb == 255, depth, 0)
        pts = self.transform(rgb, depth, cam_k)

        return pts

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    from torch.utils import data
    import matplotlib.pyplot as plt
    import os
    root = 'D:/Datasets/6dPoseData/lm'
    set = 'train'
    test_loader = data.DataLoader(BOPDataset(root, set), batch_size=8, shuffle=False)
    for bacth_id, pts in enumerate(test_loader):
        print(pts.shape)
