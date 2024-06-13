import numpy as np
import mrcfile
import torch
import trimesh
import matplotlib.pyplot as plt
import mesh_to_sdf
import sys

sys.path.append('/home/shdokan/projects/face_gen/FLAME_PyTorch/')
from flame_pytorch import FLAME, Config

import os
import cv2
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import scipy.optimize as opt
from initialize_next3d_sampling import load_next3d, LookAtPoseSampler, FOV_to_intrinsics

# Setup Dlib
import dlib
face_model_path = "./model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_model_path)

# Setup FAN model
import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

def compute_fan_points(img):
    global fa
    landmarks = fa.get_landmarks(img)
    if landmarks is None:
        return None
    if len(landmarks) < 1:
        return None
    return landmarks[0]


# define a function to generate random flame params
def generate_random_flame_params():
    # [TODO]: FLAME params between (-2, 2)
    shape_params = torch.rand([1, 100], dtype=torch.float32).cuda()*4 -2
    exp_params = torch.rand([1, 50], dtype=torch.float32).cuda()*4 -2
    pose_params = torch.zeros([1, 6], dtype=torch.float32).cuda()
    return shape_params, exp_params, pose_params

def generate_random_flame(flamelayer):
    # get random flame params first
    shape_params, exp_params, pose_params = generate_random_flame_params()
    
    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
    vertice, landmark = flamelayer(
        shape_params, exp_params, pose_params
    )
    
    if config.optimize_eyeballpose and config.optimize_neckpose:
        neck_pose = torch.zeros(1, 3).cuda()
        eye_pose = torch.zeros(1, 6).cuda()
        vertice, landmark = flamelayer(
            shape_params, exp_params, pose_params, neck_pose, eye_pose
        )
    faces = flamelayer.faces
    
    # return the mesh and landmarks
    return vertice, faces, landmark

def get_transformation_matrix(params):
    tx, ty, tz, rx, ry, rz, sx, sy, sz = params
    
    translation_matrix = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    
    rotation_matrix_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.radians(rx)), -np.sin(np.radians(rx)), 0],
        [0, np.sin(np.radians(rx)), np.cos(np.radians(rx)), 0],
        [0, 0, 0, 1]
    ])
    
    rotation_matrix_y = np.array([
        [np.cos(np.radians(ry)), 0, np.sin(np.radians(ry)), 0],
        [0, 1, 0, 0],
        [-np.sin(np.radians(ry)), 0, np.cos(np.radians(ry)), 0],
        [0, 0, 0, 1]
    ])
    
    rotation_matrix_z = np.array([
        [np.cos(np.radians(rz)), -np.sin(np.radians(rz)), 0, 0],
        [np.sin(np.radians(rz)), np.cos(np.radians(rz)), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    scaling_matrix = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])
    
    transformation_matrix = np.dot(np.dot(np.dot(translation_matrix, rotation_matrix_x), rotation_matrix_y), np.dot(rotation_matrix_z, scaling_matrix))
    
    return transformation_matrix

def project_vertices_to_image(vertices, cam2world_matrix, intrinsics, image_resolution=(64, 64)):
    """
    Project 3D vertices onto an image of given resolution.

    Parameters:
    - vertices: numpy array of shape (V, 3) representing V 3D vertices.
    - cam2world_matrix: transformation matrix (4, 4) from camera space to world space.
    - intrinsics: camera's intrinsic matrix (3, 3).
    - image_resolution: tuple (width, height) of the image resolution.

    Returns:
    - projected_coords: numpy array of shape (V, 2) representing the pixel coordinates in the image.
    """
    
    # Transform vertices to camera space
    # cam2world_pose[:3, 3] *= 0.1
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    world2cam_matrix = np.linalg.inv(cam2world_matrix)
    vertices_cam_space = (world2cam_matrix @ vertices_homogeneous.T).T[:, :3]
    
    # Project to 2D using intrinsics
    projected_homogeneous = (intrinsics @ vertices_cam_space.T).T
        
    # Convert to 2D cartesian coordinates
    projected_points = projected_homogeneous[:, :2] / projected_homogeneous[:, 2, np.newaxis]
    
    # Convert normalized coordinates to pixel coordinates
    projected_coords = np.zeros_like(projected_points)
    projected_coords[:, 0] = projected_points[:, 0] * image_resolution[0]
    projected_coords[:, 1] = projected_points[:, 1] * image_resolution[1]
    
    return projected_coords

def compute_distance_error(image, landmarks, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    dist_transform = cv2.distanceTransform(255-edges, cv2.DIST_L2, 5)
    distances = [dist_transform[y, x] for x, y in landmarks]
    valid_distances = [d for d in distances if d < threshold]
    if valid_distances:
        average_distance = np.mean(valid_distances)
    else:
        average_distance = None  # or some other indicator of no valid distances
    return average_distance

def compute_gradient_alignment_score(image, landmarks, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    dist_transform = cv2.distanceTransform(255-cv2.Canny(gray_image, 100, 200), cv2.DIST_L2, 5)
    
    angle_diffs = []
    for (x, y) in landmarks:
        distance_to_edge = dist_transform[y, x]
        if distance_to_edge < threshold:
            nearest_edge_point = np.unravel_index(np.argmin(dist_transform, axis=None), dist_transform.shape)
            dir_vector = np.array(nearest_edge_point) - np.array([y, x])
            dir_angle = np.arctan2(dir_vector[0], dir_vector[1])
            grad_dir = np.arctan2(grad_y[y, x], grad_x[y, x])
            angle_diff = np.pi - np.abs(np.pi - np.abs(grad_dir - dir_angle))
            angle_diffs.append(angle_diff)
            
    if angle_diffs:
        average_angle_diff = np.mean(angle_diffs)
    else:
        average_angle_diff = None  # or some other indicator of no valid alignment scores
    return average_angle_diff

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
    img = G.synthesis(ws, camera_params, v, noise_mode='const')['image']
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return img, camera_params

def generate_image_data():
    fov_deg = np.random.random()*8 + 12
    z = torch.randn(1, G.z_dim).to(device)

    imgs = []
    angle_p = np.random.random()*1.6 - 0.8
    angle_y = np.random.random()*1.6 - 0.8

    vertices, faces, landmarks = generate_random_flame(flamelayer)
    # print(vertices.shape, faces.shape, landmarks.shape)

    # prepare 2d landmarks
    lms = landmarks[0].cpu().numpy()
    lms += G.orth_shift.cpu().numpy()
    lms *= G.orth_scale.cpu().numpy()/2

    v = torch.cat((vertices, landmarks), 1)
    # print(v.shape)

    img, camera_params = generate_image(G, z, v, angle_p, angle_y, fov_deg)
    img = img[0]
    return img, lms, camera_params

def full_projection(transformation_matrix, img, lms, camera_params):
    intrinsic, cam2world = camera_params[0, 16:].reshape(3, 3).cpu().numpy(), camera_params[0, :16].reshape(4, 4).cpu().numpy()
    vertices_transformed = np.dot(lms, transformation_matrix[:3, :3].T) + transformation_matrix[:3, 3]
    projected_points = project_vertices_to_image(vertices_transformed, cam2world, intrinsic, img.shape[:2])
    
    return projected_points

def viz_random_projection(transformation_matrix,  plot=False):
    img, lms, camera_params = generate_image_data()
    
    # [TODO]: Do we really need this scale?
    # img = cv2.resize(img, (256, 256))
    
    projected_points = full_projection(transformation_matrix, img, lms, camera_params)
    
    if plot:
        plt.figure()
        plt.scatter(projected_points[:, 0], projected_points[:, 1], alpha=0.85, c='r', s=5)
        plt.imshow(img)
        plt.show()
    else:
        return img, projected_points

# Nfeval = 1
def aggregate_error(trans_params):
    # global Nfeval
    transformation_matrix = get_transformation_matrix(trans_params)
    # print(Nfeval)
    # Nfeval += 1
    
    error_thresh = 10
    total_error = 0
    n_runs = 20
    for ix in range(n_runs):
        # img, lms, camera_params = generate_image_data()
        
        img, projected_points = viz_random_projection(transformation_matrix)
        # only keep projected points within image boundary
        projected_points = projected_points[(projected_points[:, 0] >= 0) & (projected_points[:, 0] < img.shape[1]) & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < img.shape[0])]
        
        dist_error = compute_distance_error(img, projected_points.astype(int), error_thresh)
        if dist_error is not None:
            total_error += dist_error
            
        # align_error = compute_gradient_alignment_score(img, projected_points.astype(int), error_thresh)
        # if align_error is not None:
        #     total_error += align_error
    
    return total_error/n_runs


def compute_dlib_points(img):
    global detector, predictor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) < 1:
        return None
    landmarks = predictor(gray, faces[0])
    dlms = np.array([(ix.x, ix.y) for ix in landmarks.parts()])
    return dlms


def dlib_aggregate_error(trans_params):
    transformation_matrix = trans_params.reshape((4, 4)) #get_transformation_matrix(trans_params)
    # print(Nfeval)
    # Nfeval += 1
    total_error = 0
    n_runs = 50
    for ix in range(n_runs):
        # img, lms, camera_params = generate_image_data()
        
        img, projected_points = viz_random_projection(transformation_matrix)
        dlib_points = compute_dlib_points(img)
        
        if dlib_points is not None:
            dist_error = np.sqrt(np.sum((dlib_points - projected_points)**2))
            total_error += dist_error
    
    return total_error/n_runs

def fan_aggregate_error(trans_params):
    transformation_matrix = trans_params.reshape((4, 4)) #get_transformation_matrix(trans_params)
    # print(Nfeval)
    # Nfeval += 1
    total_error = 0
    n_runs = 200
    for ix in range(n_runs):
        # img, lms, camera_params = generate_image_data()
        
        img, projected_points = viz_random_projection(transformation_matrix)
        fan_points = compute_fan_points(img)
        
        if fan_points is not None:
            dist_error = np.sqrt(np.sum((fan_points - projected_points)**2))
            total_error += dist_error
    
    return total_error/n_runs





config = Config()
config.batch_size = 1
radian = np.pi / 180.0
flamelayer = FLAME(config)
flamelayer.cuda()

device = 'cuda'
G = load_next3d()

trans_params = np.array([0.0, 0.05, 0.14, 0.0, -3.0, -0.0, 1.0, 1.0, 1.0])
transformation_matrix = get_transformation_matrix(trans_params)
# viz_random_projection(transformation_matrix, True)

Nfeval = 1
def callback(param):
    global Nfeval
    print("{}    {}    {}".format(Nfeval, dlib_aggregate_error(param), np.round(param, 4)))
    Nfeval += 1

import datetime

# initial_param = trans_params.copy()
initial_param = transformation_matrix.copy().flatten()

start_time = datetime.datetime.now()
print(start_time)

result = opt.minimize(
    fan_aggregate_error,
    x0=initial_param,  # Your initial parameter estimate
    callback=callback,
    method='Powell',  # or 'L-BFGS-B', 'BFGS', 'Powell', etc.
    options={
        'maxiter': 100,  # Maximum number of iterations
        'disp': True,  # Display convergence messages
    },
)

print(datetime.datetime.now() - start_time)
print(result.x.reshape((4,4)))