import torch
import torch.nn as nn
import numpy as np
import smplx
import pickle
import trimesh
class DLMesh(nn.Module):
    def __init__(self, opt, num_layers_bg=2, hidden_dim_bg=16):

        super(DLMesh, self).__init__()

        self.opt = opt

        self.num_remeshing = 1


        self.device = torch.device("cuda")   
 

        self.body_model = smplx.create(
                model_path="./SMPLX_NEUTRAL_2020.npz",
                model_type='smplx',
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True,
                create_left_hand_pose=True,
                create_right_hand_pose=True,
                create_jaw_pose=True,
                create_leye_pose=True,
                create_reye_pose=True,
                create_expression=True,
                create_transl=False,
                use_pca=False,
                use_face_contour=True,
                flat_hand_mean=True,
                num_betas=300,
                num_expression_coeffs=100,
                num_pca_comps=12,
                dtype=torch.float32,
                batch_size=1,
            ).to(self.device)

        self.smplx_faces = self.body_model.faces.astype(np.int32)

        with open('./outputs/_PKL/ski/smpl.pkl', 'rb') as file:
            data = pickle.load(file)

        smplx_params = data[0]  # 提取第一个元素的参数


        self.global_orient = torch.tensor(smplx_params['smpl_parameters'][0]['global_orient'], dtype=torch.float32).squeeze(-1).to(self.device)
        # 确保形状为 [3] 或 [1, 3]
        self.global_orient = self.global_orient.view(1, -1)  # 转换为 [1, 3] 形状
        # 假设 body_pose 原始数据形状错误，我们假定为 23x3
        self.body_pose = torch.tensor(smplx_params['smpl_parameters'][0]['body_pose'], dtype=torch.float32).to(self.device)
        self.body_pose = self.body_pose.view(1, -1)  # 转换为 [1, 69]
        self.betas = torch.tensor(smplx_params['smpl_parameters'][0]['betas'], dtype=torch.float32).squeeze(-1).to(self.device)
        # 确保形状为 [10] 或 [1, 10]
        self.betas = self.betas.view(1, -1)  # 转换为 [1, 10]
        
        self.jaw_pose = torch.zeros(1, 3).to(self.device)

        self.body_pose = self.body_pose.view(-1, 3)
        self.body_pose[[0, 1, 3, 4, 6, 7], :2] *= 0
        self.body_pose = self.body_pose.view(1, -1)
        
        
        self.expression = torch.zeros(1, 100).to(self.device)
        self.faces_list, self.dense_lbs_weights, self.uniques, self.vt, self.ft = self.get_init_body()




    def generate_subdivided_smplx(self):
        output = self.body_model(
            betas=self.betas,
            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True
        )
        v_cano = output.vertices[0]  # 获取基础模型顶点坐标

        faces = self.smplx_faces  # 获取面信息

        for i in range(self.num_remeshing):
            v_cano = self.subdivide(v_cano, faces)

        return v_cano


    @torch.no_grad()
    def subdivide(self, vertices, faces, attributes=None, face_index=None):


        # Ensure vertices and attributes (if not None) are on CPU for NumPy operations
        vertices = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        faces = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
        if attributes is not None:
            attributes = attributes.cpu().numpy()

        if face_index is None:
            face_index = np.arange(len(faces))
        else:
            face_index = np.asanyarray(face_index)

        # Select faces to subdivide
        faces = faces[face_index]
        # Compute the midpoints for each triangle edge
        triangles = vertices[faces]
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])

        # Merge midpoints and create unique indices
        unique, inverse = trimesh.grouping.unique_rows(mid)
        mid = mid[unique]
        mid_idx = inverse.reshape((3, -1)).T + len(vertices)

        # Define new faces with correct winding
        f = np.column_stack([
            faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
            mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
            mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
            mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]
        ]).reshape((-1, 3))

        # Update faces
        new_faces = np.vstack((faces, f[len(face_index):]))
        new_faces[face_index] = f[:len(face_index)]

        # Add new vertices
        new_vertices = np.vstack((vertices, mid))

        # Handle attributes if provided
        if attributes is not None:
            tri_att = attributes[faces]
            mid_att = np.vstack([tri_att[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])
            mid_att = mid_att[unique]
            new_attributes = np.vstack((attributes, mid_att))
            return torch.from_numpy(new_vertices), torch.from_numpy(new_faces), torch.from_numpy(new_attributes), unique

        return torch.from_numpy(new_vertices), torch.from_numpy(new_faces), unique




opt = {
    'lock_geo': False,
    'num_betas': 10,
    'num_expression_coeffs': 10,
    'num_remeshing': 2,
    'uniques': [...]  # 提供具体的unique数组
}

dl_mesh = DLMesh(opt)
subdivided_vertices = dl_mesh.generate_subdivided_smplx()
print(subdivided_vertices)