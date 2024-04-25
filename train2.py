import pickle
import numpy as np
import torch 
import smplx
from smplx.body_models import SMPLLayer
import trimesh
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
import torchvision.transforms as T
filename = "./outputs/_DEMO/ski/img/000001.jpg"
patch_size = 14
feat_dim= 384
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')


img = Image.open(filename).convert('RGB')
H,W=img.height, img.width
patch_H,patch_W= img.height//patch_size,img.width//patch_size
print(patch_H,patch_W)
newH = H - H%patch_size
newW = W - W%patch_size
transform = T.Compose([
    T.Resize((newH, newW), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
img_tensor = transform(img)[:3].unsqueeze(0)
print(f"New widths and heights are {newW,newH}")

features = torch.zeros(1, patch_H * patch_W, feat_dim)
imgs_tensor = torch.zeros(1, 3, patch_H * patch_size, patch_W * patch_size)
imgs_tensor[0] = transform(img)[:3]
with torch.no_grad():   
    features_dict = dinov2_vits14.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']
# features = features.reshape(4 * patch_H * patch_W, feat_dim)
features=features.reshape(1,feat_dim,patch_H,patch_W)




file_path='./outputs/_PKL/run/smpl.pkl'

smpl=SMPLLayer(model_path='./SMPL_NEUTRAL.pkl').to('cuda')
with open(file_path, 'rb') as file:
    data = pickle.load(file)
with open('./SMPL_NEUTRAL.pkl', 'rb') as file:
    smpl_data = pickle.load(file, encoding='latin1')
faces=smpl_data['f']
results = []
# Process each frame's data
for frame in data:
    frame=data[210]
    frame_index = frame['frame_index'][0]
    frame_name = frame['frame_name'][0]

    global_orient = torch.tensor(frame['smpl_parameters'][0]['global_orient'], dtype=torch.float32).unsqueeze(0).to('cuda')
    body_pose = torch.tensor(frame['smpl_parameters'][0]['body_pose'], dtype=torch.float32).unsqueeze(0).to('cuda')
    betas = torch.tensor(frame['smpl_parameters'][0]['betas'], dtype=torch.float32).unsqueeze(0).to('cuda')
    translation=frame['translation'][0]
    K_matrix=frame['K_matrix'][0]
    rotation=frame['rotation'][0].to('cuda')
    _2d_joints=torch.tensor(frame['2d_joints'][0]).to('cuda')
    _2d_vertices=torch.tensor(frame['2d_vertices'][0]).to('cuda')
    output = smpl.forward(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=None,  # 假设没有平移数据，或者使用 camera_data 作为 transl 输入
            return_verts=True,
            return_full_pose=False,
            pose2rot=True
        )
    vertices=output.vertices.cpu().squeeze(0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    mesh.export('./output_mesh.obj')
    import ipdb;ipdb.set_trace()
    #2camera cooridnate
    camera_vertices = torch.einsum('bij,bkj->bki', rotation, vertices)
    camera_vertices = camera_vertices + translation.unsqueeze(1)
    projected_points = camera_vertices
    projected_points = camera_vertices / (camera_vertices[:,:,-1].unsqueeze(-1))
    #2 image coordinate
    projected_points = torch.einsum('bij,bkj->bki', K_matrix, projected_points)
    projected_points=projected_points[:, :, :-1]
    #projected_points=projected_points+0.5

    projected_points=projected_points*2
    import ipdb;ipdb.set_trace()
    print(projected_points)
    break
    # image_path = frame_name
    # img = cv2.imread(image_path)

    # if img is None:
    #     print("图像文件读取失败，请检查路径。")
    # else:
    #     height, width, _ = img.shape
    # for _ in projected_points:
    #     for point in _:


    #             x, y = point[0],point[1]

    #             x=int(x.item()*width)
    #             y=int(y.item()*height)

    #             cv2.circle(img, (x, y), 5, (0, 255, 0), 2)  


    #     cv2.imwrite("./experiment_run/" + str(frame_index) + ".jpg", img)




