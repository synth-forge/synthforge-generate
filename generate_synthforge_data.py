import numpy as np
import mrcfile
import torch
import trimesh
import matplotlib.pyplot as plt
import mesh_to_sdf
import sys
import skimage.io as sio

sys.path.append('<Path to FLAME_PyTorch>')
from flame_pytorch import FLAME, Config

import argparse
import json
from tqdm import tqdm
import skimage.io
import pickle as pkl
from collections import OrderedDict

import torch.nn.functional as F
import torch.nn as nn
import tqdm

import os
import cv2
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from initialize_next3d_sampling import load_next3d, LookAtPoseSampler, FOV_to_intrinsics, draw_landmarks

import face_alignment

from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    TexturesVertex,
    TexturesUV,
    TexturesAtlas,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    RasterizationSettings,
    AmbientLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle

colormap = OrderedDict({
    'face': (0.7, 0.3, 0.5),
    'forehead': (0.6, 0.2, 0.4),
    'eye_region': (0.8, 0.2, 0.2),
    'neck': (0.2, 0.8, 0.2),
    'right_eye_region': (0.2, 0.8, 0.8),
    'left_eye_region': (0.3, 0.5, 0.7),
    'left_eyeball': (0.2, 0.2, 0.8),
    'right_eyeball': (0.8, 0.2, 0.8),
    #'right_ear': (0.8, 0.8, 0.2),
    'scalp': (0.6, 0.4, 0.2),
    # 'boundary': (0.4, 0.2, 0.6),
    # 'left_ear': (0.5, 0.7, 0.3),
    'lips': (0.4, 0.6, 0.2),
    'nose': (0.2, 0.4, 0.6),
})


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
        img_out = G.synthesis(ws, camera_params, v, noise_mode='const')
        img, depth = img_out['image'], img_out['image_depth']
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        # normalize depth in 0-1 range
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return img, depth, camera_params

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
        
        # Setup FAN
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D, flip_input=False, device=self.device)
        
        # Setup Next3D
        self.G = load_next3d()
        # self.G.neural_rendering_resolution = 128
        self.use_cache = use_cache
        self.data_cache = {}
        
        if self.use_cache:
            self.create_cache()
        
    def create_cache(self):
        print("Creating data cache...")
        for ix in tqdm.tqdm(range(self.data_len)):
            # Generate data point
            img, depth, lms, landmarks, camera_params = self.generate_image_and_landmarks()
            
            # Keep generating until we get a valid image
            while landmarks is None:
                img, depth, lms, landmarks, camera_params = self.generate_image_and_landmarks()
            
            self.data_cache[ix] = {
                'img': img,
                'depth': depth,
                'lms': lms,
                'landmarks': landmarks,
                'camera_params': camera_params
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
        return vertice, faces, landmark, [shape_params, exp_params, pose_params]
    
    def generate_random_image(self):
        fov_deg = np.random.random()*15 + 13
        z = torch.randn(1, self.G.z_dim).to(self.device)

        imgs = []
        angle_p = np.random.random()*1.6 - 0.8
        angle_y = np.random.random()*1.6 - 0.8
        
        vertices, faces, landmarks, _ = self.generate_random_flame()
        lms = landmarks[0].cpu().numpy()
        lms += self.G.orth_shift.cpu().numpy()
        lms *= self.G.orth_scale.cpu().numpy()/2
        lms[..., 2] += np.sqrt(2)/10
        
        v = torch.cat((vertices, landmarks), 1)
        # print(v.shape)

        img, depth, camera_params = generate_image(self.G, z, v, angle_p, angle_y, fov_deg)
        img = img[0]
        return img, depth[0], lms, camera_params
    
    def generate_image_and_landmarks(self):
        img, depth, lms, camera_params = self.generate_random_image()
        landmarks = self.fa.get_landmarks(img)
        return img, depth, lms, landmarks, camera_params 
        

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if not self.use_cache:
            # Generate data point
            img, depth, lms, landmarks, camera_params = self.generate_image_and_landmarks()
            
            # Keep generating until we get a valid image
            while landmarks is None:
                img, depth, lms, landmarks, camera_params = self.generate_image_and_landmarks()
        else:
            data_idx = self.data_cache[idx]
            img, depth, lms, landmarks, camera_params = data_idx['img'], data_idx['lms'], data_idx['landmarks'], data_idx['camera_params']
        
        landmarks = landmarks[0]
        # Convert to torch tensors
        img = torch.tensor(img).permute(2, 0, 1).float()/255
        depth = torch.tensor(depth).permute(2, 0, 1).float()/255
        lms = torch.tensor(lms).float()
        landmarks = torch.tensor(landmarks).float()
        # camera_params = torch.tensor(camera_params).float()
        
        data_point = {
            'img': img.to(self.device),
            'depth': depth.to(self.device),
            'lms': lms.to(self.device),
            'landmarks': landmarks.to(self.device),
            'camera_params': camera_params[0].to(self.device)
        }
        
        return data_point

# function that takes torch variables and does 2D projection
def project_points(points_3d, camera_info, image_size):
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
    
    # Project the 3D points to 2D
    points_2d_homo = torch.mm(intrinsics, torch.mm(extrinsics, points_3d_homo.t()))
    
    # Convert back to Cartesian coordinates
    points_2d = points_2d_homo[:2, :] / points_2d_homo[2, :]
    
    # Scale by image size
    points_2d_scaled = torch.stack((points_2d[0, :] * image_size, points_2d[1, :] * image_size), dim=-1)
    
    return points_2d_scaled

def project_predictions(points_batch, camera_batch, img_size=512):
    B, N, C = points_batch.shape
    
    # Iterate over each item in batch to compute the output points
    outs = []
    for i in range(B):
        # print(points_batch[i].shape, camera_batch[i].shape)
        out = project_points(points_batch[i], camera_batch[i], img_size).unsqueeze(0)
        outs.append(out)
    
    # concatenate as torch tensor
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
        torch.nn.init.xavier_uniform_(m.weight, 0.01)
        m.bias.data.fill_(0.1)


def render_segmap(texture, verts, ffaces, fov_deg, raster_settings, device='cuda'):
    textures = texture
    vvtf = RotateAxisAngle(180, axis='Z').cuda()

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

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=light)
    )
    
    return renderer(mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Images and Annotations")
    parser.add_argument('--output_dir', type=str, default='<Default dataset path>', help='Output directory to save images and annotations')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
    args = parser.parse_args()
    
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Create images and annotations subdirectories
    images_dir = os.path.join(args.output_dir, 'images')
    annotations_dir = os.path.join(args.output_dir, 'annotations')
    depth_dir = os.path.join(args.output_dir, 'depth')
    seg_dir = os.path.join(args.output_dir, 'seg')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
    # path to the transformation model
    mlp_path = "<Path to fixMapMLP01.pth>"
    flame_color_path = '<Path to FLAME_masks.pkl>'
    device = 'cuda'
    
    # Load FLAME Texture
    ffff = np.load("<Path to FLAME_texture.npz>")
    fftex = sio.imread("<Path to flame_seg_tex.png>")
    Tmap = (torch.tensor(fftex, dtype=torch.float32)/ 255.0).unsqueeze(0).cuda()
    Tuv = torch.tensor(ffff['vt']).float().unsqueeze(0).cuda()
    Tfv = torch.Tensor(ffff['ft'].astype(np.int64)).long().unsqueeze(0).cuda()
    tex = TexturesUV(Tmap, verts_uvs=Tuv, faces_uvs=Tfv)
       
    # Setup the dataset instance
    train_dataset = ImageGenData(data_len=1)
    train_dataset.G.eval()
    net = torch.load(mlp_path)
    net.eval()
    
    # setup the color maps
    with open(flame_color_path, 'rb') as f:
        flame_colors = pkl.load(f, encoding='latin1')
        
    # Setup FLAME colors
    colors = torch.zeros(5023, 3)

    for region, indices in colormap.items():
        indices = flame_colors[region]
        color = torch.tensor(colormap[region]).view(1, 1, 3)
        # print(region, indices.max())
        for i in indices:
            colors[i, :] = color

    # Set up the renderer
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0  # Set to 0 to use naïve rasterization
    )
    
    # Setup tqdm progress bar
    pbar = tqdm.tqdm(total=args.num_images)
    i = 0
    
    while True:
        # Check exit condition
        if i >= args.num_images:
            break
        
        # Generate a single image
        fverts, ffaces, flandmarks, flame_params = train_dataset.generate_random_flame()
        shape_params, exp_params, pose_params = flame_params

        # [PARAMS] generate random z param and FOV
        z = torch.randn(1, train_dataset.G.z_dim).to(train_dataset.device)
        fov_deg = np.random.random()*6 + 18
        angle_p = np.random.random()*1.6 - 0.8
        angle_y = np.random.random()*1.6 - 0.8

        v = torch.cat((fverts, flandmarks), 1)

        # verts = fverts.clone()
        verts = v.clone()
        verts += train_dataset.G.orth_shift.cuda()
        verts *= train_dataset.G.orth_scale.cuda()/2
        verts[..., 2] += np.sqrt(2)/10
        # verts = verts.cpu().numpy()[0]
        # print(verts.shape)

        # verts = torch.FloatTensor(verts).to('cuda').unsqueeze(0)
        verts = net(verts)

        # imgs = []
        img, depth, camera_params = generate_image(train_dataset.G, z, v, angle_p, angle_y, fov_deg)
        
        # Get the actual 2D landmarks from synthetic data
        pt2d = project_predictions(verts[:, fverts.shape[1]:], camera_params)

        verts = verts[0, :fverts.shape[1]]
        points_3d_homo = torch.cat((verts, torch.ones(verts.shape[0], 1, device=verts.device)), dim=-1)

        cam2world = camera_params[:, :16].reshape(4,4)
        # Compute the extrinsic matrix [R|t] from cam2world
        world2cam = torch.inverse(cam2world)
        extrinsics = world2cam[:3, :]  # Take the first 3 rows of world2cam

        # Project the 3D points to 2D
        verts = torch.mm(extrinsics, points_3d_homo.t()).t()
        
        # Get the semantic map from the data
        seg_imgs = render_segmap(tex, verts, ffaces, fov_deg, raster_settings, device)
        segmap = seg_imgs[0, :, :, :3].detach().cpu().numpy()
        segmap = (segmap*255).astype(np.uint8)
        img = img[0]
        depth = depth[0]
        
        # compute the FAN Landmarks
        landmarks = train_dataset.fa.get_landmarks(img)
        if landmarks is None:
            continue
            # print("Error")
        landmarks = landmarks[0]
        
        # Process annotations before saving
        projected_points = pt2d.detach().cpu().numpy()[0]
        
        # Compute the error w.r.t. FAN
        mse = np.mean(np.sqrt(np.sum((landmarks - projected_points)**2, 1)))
        if mse > 11:
            continue
            # print("Error")
        
        # Gather the annotations
        annotation = {
                'z': z.cpu().numpy()[0].astype(float).tolist(),
                'camera_params': camera_params.cpu().numpy().flatten().astype(float).tolist(),
                'flame_shape': shape_params.cpu().numpy().flatten().astype(float).tolist(),
                'flame_exp': exp_params.cpu().numpy().flatten().astype(float).tolist(),
                'flame_pose': pose_params.cpu().numpy().flatten().astype(float).tolist(),
                '2d_kps': projected_points.astype(float).tolist(),
                'fan_kps': landmarks.astype(float).tolist() if landmarks is not None else 'None',
                'mse': mse.astype(float) if mse is not None else 'None',
            }
        
        # Create filenames with leading zeros, e.g., 00001.png
        filename = f"{i:05d}.png"
        image_path = os.path.join(images_dir, filename)
        depth_path = os.path.join(depth_dir, filename)
        seg_path = os.path.join(seg_dir, filename)
        annotation_path = os.path.join(annotations_dir, f"{i:05d}.json")

        # Save image, segmentation, and annotation
        skimage.io.imsave(image_path, img)
        skimage.io.imsave(seg_path, segmap)
        skimage.io.imsave(depth_path, depth[..., 0])
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f)
        
        # Update i and progress bar
        i += 1
        pbar.update(1)
    print("Process Completed...")