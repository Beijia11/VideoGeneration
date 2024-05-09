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
from PIL import Image,ImageDraw
import nvdiffrast.torch as dr
import matplotlib
import torchvision.transforms as T


img_path='./outputs/_DEMO/run/img/000001.jpg'
file_path='./outputs/_PKL/ski/smpl.pkl'
device = torch.device('cuda:1')
patch_size = 14
feat_dim= 384
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
img = Image.open(img_path).convert('RGB')
H,W=img.height, img.width
max_dimension = max(W, H)
patch_H,patch_W= img.height//patch_size,img.width//patch_size
newH = H - H%patch_size
newW = W - W%patch_size
transform = T.Compose([
    T.Resize((newH, newW), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

features = torch.zeros(1, patch_H * patch_W, feat_dim)
imgs_tensor = torch.zeros(1, 3, patch_H * patch_size, patch_W * patch_size,device=device)
imgs_tensor[0] = transform(img)[:3].unsqueeze(0).to(device)
with torch.no_grad():     
    features_dict = dinov2_vits14.forward_features(imgs_tensor)

    features = features_dict['x_norm_patchtokens']
features=features.reshape(1,feat_dim,patch_H,patch_W).to('cuda')

smpl=SMPLLayer(model_path='./SMPL_NEUTRAL.pkl').to('cuda')
with open(file_path, 'rb') as file:
    data = pickle.load(file)
with open('./SMPL_NEUTRAL.pkl', 'rb') as file:
    smpl_data = pickle.load(file, encoding='latin1')
faces=smpl_data['f'] #13776,3
faces = faces.astype(np.float16)
faces=torch.tensor(faces).to('cuda')
frame0=data[0]
_2d_vertices_0=torch.tensor(frame0['2d_vertices'][0]).to('cuda') #[0,1]
offset = ((max_dimension - W) // 2, (max_dimension - H) // 2)
_2d_vertices_0[:, 0] = (_2d_vertices_0[:, 0] * max(W,H)-offset[0])/W*2-1
_2d_vertices_0[:, 1] = _2d_vertices_0[:, 1] * max(W,H)-offset[1]/H*2-1   #[6890, 2] [-1,1]
grid = _2d_vertices_0.unsqueeze(0).unsqueeze(2)
point_features = torch.nn.functional.grid_sample(   #torch.Size([1, 384, 6890, 1])
    features,
    grid,
    mode='bilinear',  # 可以使用 'nearest' 或 'bicubic'
    padding_mode='border',  # 可以使用 'zeros' 或 'reflection'
    align_corners=True
).squeeze(3).to(device)
render_image=torch.tensor(len(data))
# Process each frame's data
for frame in data:
    frame_index = frame['frame_index'][0]
    frame_name = frame['frame_name'][0]
    global_orient = torch.tensor(frame['smpl_parameters'][0]['global_orient'], dtype=torch.float32).unsqueeze(0).to('cuda')
    body_pose = torch.tensor(frame['smpl_parameters'][0]['body_pose'], dtype=torch.float32).unsqueeze(0).to('cuda')
    betas = torch.tensor(frame['smpl_parameters'][0]['betas'], dtype=torch.float32).unsqueeze(0).to('cuda')
    translation=frame['translation'][0]
    K_matrix=frame['K_matrix'][0]
    rotation=frame['rotation'][0].to('cuda')
    _2d_vertices=torch.tensor(frame['2d_vertices'][0]).to('cuda') #[0,1]
    output = smpl.forward(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=None, 
            return_verts=True,
            return_full_pose=False,
            pose2rot=True
        )
    vertices=output.vertices.clone()
    homogeneous_vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, :1])], dim=-1).to(device)
    faces=faces.clone().to(torch.int32).to(device)
    glctx = dr.RasterizeCudaContext(device=device)
    rast_out, rast_out_db = dr.rasterize(glctx, homogeneous_vertices, faces, resolution=[newH-newH%8, newW-newW%8])
    rast_out=rast_out.to(device)
    interpolated_colors,_ = dr.interpolate(point_features, rast_out,faces)
    print(interpolated_colors)
    import ipdb;ipdb.set_trace()

    # #2camera cooridnate
    # camera_vertices = torch.einsum('bij,bkj->bki', rotation, vertices)
    # camera_vertices = camera_vertices + translation.unsqueeze(1)
    # projected_points = camera_vertices
    # projected_points = camera_vertices / (camera_vertices[:,:,-1].unsqueeze(-1))
    # #2 image coordinate
    # projected_points = torch.einsum('bij,bkj->bki', K_matrix, projected_points)
    # projected_points=projected_points[:, :, :-1]#-0.5,0.5
    # projected_points=projected_points+0.5 #0-1

    max_dimension = max(W, H)
    offset = ((max_dimension - W) // 2, (max_dimension - H) // 2)
    _2d_vertices[:, 0] = (_2d_vertices[:, 0] * max(W,H)-offset[0])/W*newW 
    _2d_vertices[:, 1] = _2d_vertices[:, 1] * max(W,H)-offset[1]/H*newH   #[6890, 2]
    img = T.ToPILsImage()(img_tensor.squeeze()).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x, y in _2d_vertices:
        # 在x, y位置画点，这里直接在张量上修改值
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='red')
    img.save("./haha.png")
    break

    
  




