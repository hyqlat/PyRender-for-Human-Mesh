import io
import os
import os.path as osp

import cv2
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.animation import FuncAnimation, writers
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from scipy.spatial.transform import Rotation as st


#import src.util.render_utils as vis_util
#from src.util.common import resize_img
import PIL.Image as Image

import threading

from classes.utils.utils import plot_cylinder, plot_point


class PyRenSkeleton:
    def __init__(self, opt, color, draw_lines):
        '''
        input:
            opt:input_dicts [Dict]
            color:color for each points [List, C]
            draw_lines:the line between certain points you want to draw [List of tuple, XX * 2]
        '''
        self.opt = opt
        self.color = color
        self.draw_lines = draw_lines
        self.img_size = self.opt.get('img_size', 5.12)
        self.res_size = self.opt.get('res_size',(512, 512))


    def draw_pts(self, ax, points, frame_num=-1):
        radius_point=self.opt.get("radius_point", 60)
        radius_line=self.opt.get("radius_line", 20)
        
        color_temp = 'k' if frame_num < self.opt["t_his"] else '#54B345' #'#8FC486'
        
        for i in range(len(points)):
            plot_point(ax, point=points[i], color=color_temp, radius=radius_point)       
        
        cou = 0
        for line in self.draw_lines:
            a, b = line[0], line[1]
            
            color_temp = 'gray' if frame_num < self.opt["t_his"] else self.color[cou]
            plot_cylinder(ax, points[a], points[b], color_temp, radius=radius_line)
            
            #ax.plot_surface(rotated_cylinder_points[..., 0], rotated_cylinder_points[..., 1], rotated_cylinder_points[..., 2], color=color[cou])
            cou += 1

    def gene_oneframe_total(self, pred, x_min, x_max, frame_num):
        #C 3
        fig = plt.figure(figsize=(self.img_size, self.img_size))
        
        elev = self.opt.get('elev', 1)
        azim = self.opt.get('azim', 20)
        
        #画模型预测结果1
        #ax[2].imshow(pic_alpha)
        ax1 = fig.add_subplot(111, projection = '3d')
        self.draw_pts(ax1, pred, frame_num)
        ax1.set_title('predict result')
        ax1.view_init(elev=elev, azim=azim)
        #smin = np.min(gt)
        #smax = np.max(gt)
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([x_min, x_max])
        ax1.set_zlim([x_min, x_max])
        ax1.axis('off')
        
        fig.subplots_adjust(wspace=0.2, hspace=0)
        #plt.plot()
        #plt.show()
        plt.tight_layout()
        
        return fig


    def render_frame(self, p3d_draw, x_min, x_max, i, file_path):
        fig = self.gene_oneframe_total(p3d_draw[i], x_min, x_max, i)
        fig.savefig(file_path + '/picture/fig{}.jpg'.format(i))

    def render_videos(self, p3d_draw):
        '''
        input:
            p3d_draw: skeleton of human with shape (T, N, 3)
        '''
        # Your code to create directories and set up video writer

        file_path = self.opt.get('save_path', 'output')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path2 = file_path + '/picture'
        if not os.path.exists(file_path2):
            os.makedirs(file_path2)
        T, C, _ = p3d_draw.shape
        
        x_min = np.min(p3d_draw)
        x_max = np.max(p3d_draw)

        threads = []
        for i in range(T):
            thread = threading.Thread(target=self.render_frame, args=(p3d_draw, x_min, x_max, i, file_path))
            threads.append(thread)
            # thread.start()
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()    
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videowrite = cv2.VideoWriter(file_path + '/result_video.avi', fourcc, self.opt.get('fps', 30), self.res_size)#FPS是帧率，size是图片尺寸

        for filename in [file_path + '/picture/fig{}.jpg'.format(i) for i in range(T)]:#这个循环是为了读取所有要用的图片文件
            img = cv2.imread(filename)
            #print(filename)
            if img is None:
                print(filename + " is error!")
                continue
            videowrite.write(img)
        videowrite.release()