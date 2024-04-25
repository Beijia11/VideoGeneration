import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
import torchvision.transforms as T
'''
dinov2提取特征
'''
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


'''
读取smpl情况，把3d点转化成2d的点位置
'''

