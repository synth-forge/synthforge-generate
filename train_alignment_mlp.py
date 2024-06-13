import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append('<Path to FLAME_PyTorch>')
from flame_pytorch import FLAME, Config

import os
import cv2
import copy
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from initialize_next3d_sampling import load_next3d, LookAtPoseSampler, FOV_to_intrinsics, draw_landmarks

import face_alignment

import torch.nn.functional as F
import torch.nn as nn
import tqdm
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    TexturesVertex,
    MeshRasterizer,
    HardPhongShader,
    RasterizationSettings,
    AmbientLights,
    MeshRendererWithFragments
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle


def generate_image(G, z, v, angle_p, angle_y, fov_deg=18.86, truncation_psi=0.7, truncation_cutoff=14, device='cuda'):
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)


        # conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = cam2world_pose.clone()
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        gan_out = G.synthesis(ws, camera_params, v, noise_mode='const')
        img = gan_out['image']
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        depth_image = gan_out['image_depth'].permute(0, 2, 3, 1)[..., 0].cpu().numpy()
        return img, camera_params, gan_out, depth_image


# Create a torch dataset class
class ImageGenData(torch.utils.data.Dataset):
    def __init__(self, data_len=200, use_cache=True):
        self.device = 'cuda'
        self.data_len = data_len
        
        # Setup FLAME
        self.config = Config()
        self.config.batch_size = 1
        self.radian = np.pi / 180.0
        self.flamelayer = FLAME(self.config)
        self.flamelayer.cuda()
        
        self.transformation_matrix = np.array([[ 9.98630006e-01,  2.08295391e-07, -5.23355055e-02, -2.58440067e-07],
                                                [ 2.56990461e-07,  9.99999919e-01,  4.26312741e-07,  5.00002752e-02],
                                                [ 5.23358776e-02,  5.17414525e-08,  9.98629752e-01,  1.40000042e-01],
                                                [ 3.37515539e-07,  1.83355493e-07,  2.68736360e-07,  1.00000021e+00],])
        
        # Setup FAN
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D, flip_input=False, device=self.device)
        
        # Setup Next3D
        self.G = load_next3d()
        self.G.neural_rendering_resolution = 128
        self.use_cache = use_cache
        self.data_cache = {}
        
        self.fov_min = 15
        self.fov_max = 23
        self.faces = None
        
        if self.use_cache:
            self.create_cache()
        
    def create_cache(self):
        print("Creating data cache...")
        for ix in tqdm.tqdm(range(self.data_len)):
            # Generate data point
            img, lms, landmarks, camera_params, vms, gan_out, depth, fov_deg = self.generate_image_and_landmarks()
            
            # Keep generating until we get a valid image
            while landmarks is None:
                img, lms, landmarks, camera_params, vms, gan_out, depth, fov_deg = self.generate_image_and_landmarks()
            
            self.data_cache[ix] = {
                'img': img,
                'lms': lms,
                'landmarks': landmarks,
                'camera_params': camera_params,
                'verts': vms,
                'nerf_pts': gan_out['nerf_pts'],
                'sigmas': gan_out['nerf_out'],
                'depth': depth,
                'fov_deg': fov_deg
            }
    
    # define a function to generate random flame params
    def generate_random_flame_params(self):
        shape_params = torch.rand([1, 100], dtype=torch.float32).cuda()*4 -2
        exp_params = torch.rand([1, 50], dtype=torch.float32).cuda()*4 -2
        pose_params = torch.zeros([1, 6], dtype=torch.float32).cuda()
        return shape_params, exp_params, pose_params
    
    def generate_random_flame(self):
        # get random flame params first
        shape_params, exp_params, pose_params = self.generate_random_flame_params()
        
        # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
        vertice, landmark = self.flamelayer(
            shape_params, exp_params, pose_params
        )
        
        if self.config.optimize_eyeballpose and self.config.optimize_neckpose:
            neck_pose = torch.zeros(1, 3).cuda()
            eye_pose = torch.zeros(1, 6).cuda()
            vertice, landmark = self.flamelayer(
                shape_params, exp_params, pose_params, neck_pose, eye_pose
            )
        faces = self.flamelayer.faces
        
        # return the mesh and landmarks
        return vertice, faces, landmark
    
    def generate_random_image(self):
        fov_deg = np.random.random()*(self.fov_max - self.fov_min) + self.fov_min
        z = torch.randn(1, self.G.z_dim).to(self.device)

        imgs = []
        angle_p = (np.random.random()*2 - 1)*0.5
        angle_y = (np.random.random()*2 - 1)*0.5
        
        vertices, faces, landmarks = self.generate_random_flame()
        if self.faces is None:
            self.faces = faces
        lms = landmarks[0].cpu().numpy()
        lms += self.G.orth_shift.cpu().numpy()
        lms *= self.G.orth_scale.cpu().numpy()/2
        lms[..., 2] += np.sqrt(2)/10
        
        # lms = np.dot(lms, self.transformation_matrix[:3, :3].T) + self.transformation_matrix[:3, 3]
        
        v = torch.cat((vertices, landmarks), 1)
        # print(v.shape)

        img, camera_params, gan_out, depth = generate_image(self.G, z, v, angle_p, angle_y, fov_deg)
        img = img[0]
        
        # Update the vertices as well
        vms = vertices[0].cpu().numpy()
        vms += self.G.orth_shift.cpu().numpy()
        vms *= self.G.orth_scale.cpu().numpy()/2
        vms[..., 2] += np.sqrt(2)/10
        
        return img, lms, camera_params, vms, gan_out, depth, fov_deg
    
    def generate_image_and_landmarks(self):
        img, lms, camera_params, vms, gan_out, depth, fov_deg = self.generate_random_image()
        landmarks = self.fa.get_landmarks(img)
        return img, lms, landmarks, camera_params, vms, gan_out, depth, fov_deg
        

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if not self.use_cache:
            # Generate data point
            img, lms, landmarks, camera_params, vms, gan_out, depth, fov_deg = self.generate_image_and_landmarks()
            
            # Keep generating until we get a valid image
            while landmarks is None:
                img, lms, landmarks, camera_params, vms, gan_out, depth, fov_deg = self.generate_image_and_landmarks()
            # get nerf points and data
            nerf_pts = gan_out['nerf_pts']
            sigmas = gan_out['nerf_out']
        else:
            data_idx = self.data_cache[idx]
            img, lms, landmarks, camera_params, vms, nerf_pts, sigmas, depth = data_idx['img'], data_idx['lms'], data_idx['landmarks'], data_idx['camera_params'], data_idx['verts'], data_idx['nerf_pts'], data_idx['sigmas'], data_idx['depth']
            fov_deg = data_idx['fov_deg']
            
        landmarks = landmarks[0]
        # Convert to torch tensors
        img = torch.tensor(img).permute(2, 0, 1).float()/255
        lms = torch.tensor(lms).float()
        landmarks = torch.tensor(landmarks).float()
        # camera_params = torch.tensor(camera_params).float()
        vms = torch.tensor(vms).float()
        depth = torch.tensor(depth).float()
        fov_deg = torch.tensor([fov_deg])
        
        data_point = {
            'img': img.to(self.device),
            'lms': lms.to(self.device),
            'landmarks': landmarks.to(self.device),
            'camera_params': camera_params[0].to(self.device),
            'verts': vms.to(self.device),
            'nerf_pts': nerf_pts.to(self.device)[0],
            'sigmas': sigmas['sigma'].to(self.device)[0],
            'depth': depth.to(self.device),
            'fov_deg': fov_deg
        }
        
        return data_point

# function that takes torch variables and does 2D projection
def project_points(points_3d, camera_info, image_size, return_3d=False):
    # Assume points_3d is of shape (N, 3), cam2world is of shape (4, 4),
    # intrinsics is of shape (3, 3), and image_size is a tuple (height, width).
    cam2world, intrinsics = camera_info[:16], camera_info[16:]
    cam2world = cam2world.reshape(4, 4)
    intrinsics = intrinsics.reshape(3, 3)
    
    # Convert points_3d to homogeneous coordinates
    points_3d_homo = torch.cat((points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)), dim=-1)
    
    # Compute the extrinsic matrix [R|t] from cam2world
    world2cam = torch.inverse(cam2world)
    extrinsics = world2cam[:3, :]  # Take the first 3 rows of world2cam
    
    # 3d points
    proj_3d_points = torch.mm(extrinsics, points_3d_homo.t())
    
    # Project the 3D points to 2D
    points_2d_homo = torch.mm(intrinsics, proj_3d_points)
    
    # Convert back to Cartesian coordinates
    points_2d = points_2d_homo[:2, :] / points_2d_homo[2, :]
    
    # Scale by image size
    points_2d_scaled = torch.stack((points_2d[0, :] * image_size, points_2d[1, :] * image_size), dim=-1)
    
    if return_3d:
        return points_2d_scaled, proj_3d_points
    return points_2d_scaled

def project_predictions(points_batch, camera_batch, img_size=512, return_3d=False):
    B, N, C = points_batch.shape
    
    # Iterate over each item in batch to compute the output points
    outs = []
    points_3d = []
    for i in range(B):
        # print(points_batch[i].shape, camera_batch[i].shape)
        outs_ = project_points(points_batch[i], camera_batch[i], img_size, return_3d)
        if return_3d:
            out, out_3d = outs_[0], outs_[1].t()
            points_3d.append(out_3d.unsqueeze(0))
        else:
            out = outs_
        outs.append(out.unsqueeze(0))
    
    # concatenate as torch tensor
    if return_3d:
        return torch.cat(outs, 0), torch.cat(points_3d, 0)
    return torch.cat(outs, 0)

# create a model
class MLP(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B*N, C)
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = torch.selu(x)
        x = self.fc2(x)
        x = x.view(B, N, -1)
        return x[..., :3]

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        # Initialize weights as close to the identity matrix as possible
        identity_matrix = torch.eye(m.weight.shape[0], m.weight.shape[1])
        m.weight.data.copy_(identity_matrix)

        # Add small jitter noise to the weights
        noise_stddev = 0.01  # Adjust this value as needed
        # torch.nn.init.normal_(m.weight.data, mean=0, std=noise_stddev)
        # torch.nn.init.xavier_uniform_(m.weight, 0.01)
        m.bias.data.fill_(0.01)


def render_depthmap(verts, ffaces, fov_deg, device='cuda'):
    raster_settings = RasterizationSettings(
        image_size=128, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0  # Set to 0 to use naïve rasterization
    )
    vvtf = RotateAxisAngle(180, axis='Z').cuda()
    textures = TexturesVertex(torch.zeros_like(verts).cuda().unsqueeze(0))

    new_verts = vvtf.transform_points(verts.to(device))

    mesh = Meshes(
        verts=[new_verts.to(device)], 
        faces=[torch.tensor(ffaces.astype(np.int64)).to(device)],
        textures=textures
    )

    extrinsics = torch.eye(4).cuda()
    R, t = extrinsics[:3, :3], extrinsics.reshape((4,4))[:3, 3]
    cameras = FoVPerspectiveCameras(device=device, fov=fov_deg/1.414, R=R.unsqueeze(0), T=-t.unsqueeze(0))

    light = AmbientLights(device=device)
    
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras)#, lights=light)
    )
    
    return renderer(mesh)[1].zbuf

def compute_nme(predictions, ground_truth, left_eye_idx=36, right_eye_idx=45):
    """
    Compute Normalized Mean Error (NME) between predicted and ground truth landmarks.
    
    Args:
        predictions (torch.Tensor): Predicted landmarks of shape (B, N, 2).
        ground_truth (torch.Tensor): Ground truth landmarks of shape (B, N, 2).
        left_eye_idx (int): The index of the left eye corner in the landmarks.
        right_eye_idx (int): The index of the right eye corner in the landmarks.
        
    Returns:
        nme (torch.Tensor): The Normalized Mean Error for each example in the batch.
    """
    # Compute the euclidean distance between predicted and ground truth landmarks
    error = torch.norm(predictions - ground_truth, dim=-1)
    
    # Compute the inter-ocular distance for each example in the batch
    inter_ocular_distance = torch.norm(
        ground_truth[:, left_eye_idx] - ground_truth[:, right_eye_idx],
        dim=-1
    )
    
    # Normalize the error by the inter-ocular distance
    normalized_error = error / inter_ocular_distance.unsqueeze(1).expand_as(error)
    
    # Compute the mean error across landmarks for each example in the batch
    nme = normalized_error.mean(dim=1)
    
    return nme


if __name__ == '__main__':
    # Get the data
    train_dataset = ImageGenData(data_len=50)
    val_dataset = ImageGenData(data_len=100, use_cache=True)

    # prepare data loaders
    _, train_dataset.faces, _ = train_dataset.generate_random_flame()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)

    # Create a network instance
    net = MLP(hidden_size=4).cuda()
    net.apply(init_weights_xavier)
    
    # setup training data and optimizers
    train_dataset.use_cache = False
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-03)

    best_val = 100
    best_model = None

    num_epochs = 10
    train_loader.dataset.use_cache = False
    val_loader.dataset.use_cache = True
    loss_train = []
    loss_val = []

    # placeholder to track best loss
    prev_loss = 500

    # Train the model
    for epoch in range(num_epochs):
        net.train()
        # Iterate over the batches
        train_loss = 0
        for i, batch in enumerate(train_loader):
            # Get the data
            img = batch['img']
            lms = batch['lms']
            camera_params = batch['camera_params']
            gt_points = batch['landmarks']
            vtr_batch, pts_gt_batch, sigma_gt_batch = batch['verts'], batch['nerf_pts'], batch['sigmas']
            # print(vtr_batch.shape, pts_gt_batch.shape, sigma_gt_batch.shape)
            
            chamfer_loss = 0
            depth_loss = 0
            vtr_pred = net(vtr_batch)
            
            # Compute the 3d points in camera space
            _, vts_3d = project_predictions(vtr_pred, camera_params, return_3d=True)
            
            for bx in range(vtr_batch.shape[0]):
                vtr, pts_gt, sigma_gt = vtr_pred[bx], pts_gt_batch[bx], sigma_gt_batch[bx]
                
                sigma_gt -= sigma_gt.min()
                sigma_gt /= sigma_gt.max()
                flame_mask = (vtr_batch[bx, ..., 2] > 0.09)*(vtr_batch[bx, ..., 1] > -0.3)
                thresh = 0.35#sorted(sigma_gt, reverse=True)[10000]
                nerf_mask = (sigma_gt > 0.4).flatten() * (pts_gt[..., 2] > 0.1).flatten()
                # print((pts_gt[..., 2] > 0.1).shape, sigma_gt.shape, nerf_mask.shape)
                pc1, pc2 = vtr[flame_mask].unsqueeze(0), pts_gt[nerf_mask, :].unsqueeze(0)
                # print(pc1.requires_grad, pc2.requires_grad)
                
                chamfer_dist = chamfer_distance(pc1, pc2)
                chamfer_loss += chamfer_dist[0]*300
                
                # Now compute pred depth maps inside the batchwise loop
                vtx_bx, fov_val = vts_3d[bx], batch['fov_deg'][bx]
                pred_depth = render_depthmap(vtx_bx, train_dataset.faces, fov_val)[0, ..., 0]
                depth_gt = batch['depth'][bx, 0]
                pred_dep_mask = (pred_depth > -1).detach()
                pred_value, gt_value = pred_depth*pred_dep_mask, depth_gt*pred_dep_mask
                depth_error = F.mse_loss(pred_value.unsqueeze(0), gt_value.unsqueeze(0), reduction='mean')
                
                depth_loss += depth_error*1000
            # print(chamfer_loss)
            
            loss = chamfer_loss + depth_loss
            # print(chamfer_loss, depth_loss)
            # Forward pass on the points
            updated_points = net(lms)
            # Compute the projections
            pred = project_predictions(updated_points, camera_params)
            
            # Repeat the loss for extra points
            selected = list(range(17)) + [33]
            extra_points_gt = gt_points[:, selected, :]
            extra_points_pred = pred[:, selected, :]
            
            # Compute the loss and extra loss
            kp_loss = F.mse_loss(pred, gt_points, reduction='mean') / 5
            extra_kp_loss = F.smooth_l1_loss(extra_points_pred, extra_points_gt, reduction='mean') * 1.2
            #print(chamfer_loss.item(), depth_loss.item(), kp_loss.item(), extra_kp_loss.item())
            loss += kp_loss + extra_kp_loss
            train_loss += loss.item()
            
            # Backprop and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # # Print the loss
            # if i % 100 == 0:
            #     print(f'Epoch: {epoch}, Iter: {i}, Batch: {i}, Loss: {loss.item()}')
        # # Refresh dataset every few epochs
        # if (epoch+1) % 25 == 0:
        #     train_loader.dataset.create_cache()
        loss_train.append(train_loss/len(train_dataset))
        
        # Evaluate on the validation set
        val_loss = 0
        with torch.no_grad():
            net.eval()
            for i, batch in enumerate(val_loader):
                # Get the data
                img = batch['img']
                lms = batch['lms']
                camera_params = batch['camera_params']
                gt_points = batch['landmarks']

                # Forward pass on the points
                updated_points = net(lms)
                # Compute the projections
                pred = project_predictions(updated_points, camera_params)

                # Compute the loss
                # loss = F.mse_loss(pred, gt_points, reduction='mean')
                loss = compute_nme(pred, gt_points).mean() * 100
                # update val_loss
                val_loss += loss.item()
                
                # # Print the loss
                # if i % 100 == 0:
                #     print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
        
        # print val loss in this epoch
        print(f'Epoch: {epoch}, Val Loss: {val_loss/len(val_dataset)}, Train Loss: {train_loss/len(train_dataset)}')
        loss_val.append(val_loss/len(val_dataset))
        
        if epoch == 10:
            optimizer.param_groups[0]['lr'] *= 0.1
        
        # stopping criterion on tolerance
        current_loss = val_loss/len(val_dataset)
        difference = np.abs(prev_loss - current_loss)
        if difference < 3e-5:
            break
        prev_loss = current_loss
        
        if loss_val[-1] < best_val:
            best_model = copy.deepcopy(net)
            best_val = loss_val[-1]
    
    # save the best model      
    torch.save(best_model, "./fixMapMLP01.pth")