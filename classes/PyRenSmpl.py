import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import smplx
from classes.meshviewer import Mesh, MeshViewer, colors

from classes.utils.utils import params2torch
from classes.utils.utils import to_cpu

import trimesh
import cv2
from pyrender.constants import RenderFlags

class PyRenSmpl:
    def __init__(self, input_dicts) -> None:
        '''
        input_dicts:
        ### a dict contains ###
        smpl_model_path: path to smpl model
        save_path: path to save output result
        mv_width: meshviewer width      float
        mv_height: meshviewer height    float
        bg_color: background color     float(4)
        device: device

        camera_dict: dict of camera
            ca_rotate:camera rotate angle   float(3)[80, 0, 30]
            ca_translate: camera translate position   float(3)[1.2, -2.3, 0.5]
        
        light_dict: dict of light
            li_rotate:light rotate angle   float(3)[50, 0, 10]
            li_translate: light translate position   float(3)[0, 0, 1]
            li_intens:light intensity                   float  3.0

        render_ground:need a ground?        bool
        gr_translate: ground translate      float(3)
        gr_extent:ground size               float(3)
        gr_color:ground color               float
        ###                 ###
        '''
        self.smpl_model_path = input_dicts['smpl_model_path']
        self.save_path = input_dicts['save_path']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.mv_width = input_dicts['mv_width']
        self.mv_height = input_dicts['mv_height']
        self.bg_color = input_dicts['bg_color']#[1.0, 1.0, 1.0, 1.0]
        self.device = input_dicts['device']

        #my scene
        self.mv = MeshViewer(
            offscreen=True, 
            width=self.mv_width, 
            height=self.mv_height,
            bg_color=self.bg_color,
            camera_dict=input_dicts['camera_dict'],
            light_dict=input_dicts['light_dict']
        )

        # set the camera pose
        #x:横着的轴，往右正方向 y:前后的轴，往前正方向 z:竖着的轴，向上正方向；旋转都是正方向时，正值为顺时针；注意负值旋转相当于360+该负值

        #ground
        self.has_ground = input_dicts['render_ground']
        if self.has_ground:
            self.gr_translate = input_dicts['gr_translate']#[0, 0, -1.02]
            self.gr_extent = input_dicts['gr_extent']#(20,12,0.1)
            self.gr_color = input_dicts['gr_color']
            #
            ground_pose = np.eye(4)
            ground_pose[:3, 3] = np.array(self.gr_translate)#[-.6, -2.4, .3] ##x：左右 y z:高度
            self.gr_mesh = trimesh.creation.box(extents=self.gr_extent, transform=ground_pose)
            # t_mesh_color = np.random.uniform(size=t_mesh.faces.shape)
            t_mesh_color = np.ones(shape=self.gr_mesh.faces.shape) * self.gr_color
            self.gr_mesh.visual.face_colors = t_mesh_color
    
    def render_videos(
            self,
            sequence, 
            key,
            his_frame=10, 
            smpl_model='smpl'
        ):
        '''
        sequence: dict with 'poses' ('betas') ('gender')
        key:output file name
        his_frame:his_frame
        smpl_model: smpl smpl-x smpl-h
        '''

        seq_data = sequence['poses']
        gender = sequence.get('gender','male') #str(sequence['gender'])
        T = seq_data.shape[0]

        sbj_m = smplx.create(model_path=self.smpl_model_path,
                            model_type=smpl_model,
                            gender=gender,
                            # num_pca_comps=24,
                            # v_template=sbj_vtemp,
                            use_pca=False,
                            batch_size=T,
                            ext='pkl').to(self.device)
        if smpl_model != 'smpl':
            sbj_m.pose_mean[:] = 0
        sbj_m.use_pca = False
        sbj_params = {}

        global_rot = np.zeros_like(seq_data[:, :3])
        global_rot[:] = np.array([1.5, 0, 0])
        sbj_params['global_orient'] = global_rot
        # sbj_params['global_orient'] = seq_data[:, :3]

        # if w_golbalrot:
        if smpl_model == 'smplh':
            sbj_params['body_pose'] = seq_data[:, 3:66]
            # sbj_params['jaw_pose'] = seq_data[:, 66:69]
            # sbj_params['leye_pose'] = seq_data[:, 69:72]
            # sbj_params['reye_pose'] = seq_data[:, 72:75]
            sbj_params['left_hand_pose'] = seq_data[:, 66:111]
            sbj_params['right_hand_pose'] = seq_data[:, 111:156]
            # sbj_params['transl'] = sequence['trans']
        elif smpl_model == 'smplx':
            sbj_params['body_pose'] = seq_data[:, 3:66]
            sbj_params['jaw_pose'] = seq_data[:, 66:69]
            sbj_params['leye_pose'] = seq_data[:, 69:72]
            sbj_params['reye_pose'] = seq_data[:, 72:75]
            sbj_params['left_hand_pose'] = seq_data[:, 75:120]
            sbj_params['right_hand_pose'] = seq_data[:, 120:165]
            sbj_params['expression'] = np.zeros_like(seq_data[:,:10])
        elif smpl_model == 'smpl':
            sbj_params['body_pose'] = seq_data[:, 3:72]


        sbj_params['betas'] = sequence.get('betas', np.random.randn(10))[None, :10]

        sbj_parms = params2torch(sbj_params, device=self.device)
        verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)#72+10参数生成6890顶点

        skip_frame = 1
        imgs = []
        for frame in range(0, T, skip_frame):
            plt.cla()
            if frame < his_frame:
                col = colors['pink']
            else:
                col = colors['orange']
            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=col, smooth=True)
            # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

            # t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

            # mv.set_static_meshes([o_mesh, s_mesh, t_mesh])
            if self.has_ground:
                self.mv.set_static_meshes([s_mesh, self.gr_mesh])
            else:
                self.mv.set_static_meshes([s_mesh])

            flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
            col, _ = self.mv.viewer.render(self.mv.scene, flags=flags)#mv是MeshViewer类，viewer是pyrender.Viewer
            imgs.append(col[:, :, [2, 1, 0]])
        
        #生成视频
        
        video_name = f'{self.save_path}/{key}.avi'
        # images = [img for img in os.listdir(path_tmp) if img.endswith(".jpg")]
        # frame = cv2.imread(os.path.join(path_tmp, images[0]))
        height, width, layers = imgs[0].shape

        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_name, fourcc, frameSize=(width, height), fps=30)
        for image in imgs:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()
        



    