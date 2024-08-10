# Copyright (C) 2023 OPPO. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
import os
import datetime
import logging
sys.path.insert(0, os.path.abspath('./'))
from numpy.lib.stride_tricks import broadcast_to
from src.load_llfff import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
import cv2
import glob
import shutil
import pandas as pd
from utils import eval_uvst, rm_folder, AverageMeter, rm_folder_keep, eval_trans
from src.cam_view import rayPlaneInter, get_rays, rotation_matrix_from_vectors
from src.model import Nerf4D_relu_ps
from src.model_addon2 import Nerf4D_relu_ps_addon2
from src.model_addon2_2 import Nerf4D_relu_ps_addon2_2
from src.model_addon4 import Nerf4D_relu_ps_addon4
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--batch_size',type=int, default = 8192,help='normalize input')
parser.add_argument('--test_freq',type=int,default=25,help='test frequency')
parser.add_argument('--save_checkpoints',type=int,default=20,help='checkpoint frequency')
parser.add_argument('--whole_epoch',type=int,default=120,help='checkpoint frequency')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument("--factor", type=int, default=1, help='downsample factor for LLFF images')
parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
parser.add_argument('--norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--st_norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--work_num',type=int, default= 15, help= "normalize the data uvst")
parser.add_argument('--lr_pluser',type=int, default = 100,help = 'scale for dir')
parser.add_argument('--lr',type=float,default=5e-04,help='learning rate')
parser.add_argument('--loadcheckpoints', action='store_true', default = False)
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 256)
## addon
parser.add_argument('--load_exp', type=str, default = None,help = 'load exp name')
parser.add_argument('--addon', type=int, default= 0, help= "addon mode")
parser.add_argument('--loadckptnum',type=int, default = 0,help = 'epoch number to load')
parser.add_argument('--val', action='store_true', default = False)

class train():
    def __init__(self,args):
         # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        # data root
        data_root = args.data_dir
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.factor)) 
        if args.addon == 2: 
            self.model = Nerf4D_relu_ps_addon2(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        elif args.addon == 22: 
            self.model = Nerf4D_relu_ps_addon2_2(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        elif args.addon == 4: 
            self.model = Nerf4D_relu_ps_addon4(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        else: self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        for name, module in self.model.named_children():
                print(name, module)
        # uv and st depth
        self.uv_depth = args.uv_depth
        # normalize factor
        self.norm_fac = args.norm_fac
        self.st_norm_fac = args.st_norm_fac
        self.exp = 'Exp_'+args.exp_name
        self.exn = args.exp_name
        # tensorboard writer
        self.summary_writer = SummaryWriter("src/tensorboard/"+self.exp)
        # save img
        self.save_img = True
        # self.img_folder_train = 'result/'+self.exp+'/train/'
        self.img_folder_test = 'result/'+self.exp+'/test/'
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'
        # make folder
        # rm_folder(self.img_folder_train)
        rm_folder(self.img_folder_test)
        rm_folder_keep(self.checkpoints)
        handlers = [logging.StreamHandler()]
        dt_string = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler('result/'+self.exp+f'/{dt_string}.log', mode='w'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S', handlers=handlers,
        )
        if args.load_exp is not None:
            # load_ckpt
            self.loadexp = 'Exp_'+args.load_exp
            self.loadckpt = 'result/'+self.loadexp+'/checkpoints/'
        # load log and lr
            self.load_exp_ckpt(args.loadckptnum)
        # load checkpoints
        if(args.loadcheckpoints):
            self.load_check_points()
        self.model = self.model.cuda()
        # height and width
        image_paths = glob.glob(f"{data_img}/*.png")
        sample_img = cv2.imread(image_paths[0])
        self.h = int(sample_img.shape[0]/args.img_scale)
        self.w = int(sample_img.shape[1]/args.img_scale)
        self.img_scale = args.img_scale
        self.step_count = 0
        # load nelf data
        print(f"Start loading...")
        split='train'
        # center
        self.uvst_whole   = np.load(f"{data_root}/uvst{split}.npy") / args.norm_fac
        self.uvst_whole[:,2:] /= self.st_norm_fac
        # norm to 0 to 1
        self.uvst_min = self.uvst_whole.min()
        self.uvst_max = self.uvst_whole.max()
        self.uvst_whole = (self.uvst_whole - self.uvst_min) / (self.uvst_max - self.uvst_min) * 2 - 1.0
        # center color
        self.color_whole   = np.load(f"{data_root}/rgb{split}.npy")
        self.trans        = np.load(f"{data_root}/trans{split}.npy")
        self.intrinsic    = np.load(f"{data_root}/k{split}.npy")
        self.fdepth       = np.load(f"{data_root}/fdepth{split}.npy") # center object
        self.render_pose  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
        self.st_depth     = -self.fdepth
        self.uvst_whole = np.concatenate([self.uvst_whole]*args.rep, axis=0)
        self.color_whole = np.concatenate([self.color_whole]*args.rep, axis=0)
        split='val'
        self.uvst_whole_val   = np.load(f"{data_root}/uvst{split}.npy") / args.norm_fac
        self.uvst_whole_val[:,2:] /= self.st_norm_fac
        self.color_whole_val   = np.load(f"{data_root}/rgb{split}.npy")
        self.uvst_whole_val = (self.uvst_whole_val - self.uvst_min) / (self.uvst_max - self.uvst_min) * 2 - 1.0
        self.trans_val        = np.load(f"{data_root}/trans{split}.npy")
        self.intrinsic_val    = np.load(f"{data_root}/k{split}.npy")
        self.fdepth_val       = np.load(f"{data_root}/fdepth{split}.npy") # center object
        self.render_pose_val  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
        self.st_depth_val     = -self.fdepth
        print("Stop loading...")
        rays_whole = np.concatenate([self.uvst_whole, self.color_whole], axis=1)
        self.min_u,self.max_u,self.min_v,self.max_v,self.min_s,self.max_s,self.min_t,self.max_t = eval_uvst(rays_whole)
        self.max_x,self.min_x,self.max_y,self.min_y = eval_trans(self.trans)
        logging.info(f"u[{self.min_u},{self.max_u}], v[{self.min_v}, {self.max_v}], s[{self.min_s}, {self.max_s}], t[{self.min_t}, {self.max_t}]")
        logging.info(f"x[{self.min_x},{self.max_x}], y[{self.min_y}, {self.max_y}]")
        # data loader
        self.uvst_whole_gpu    = torch.tensor(self.uvst_whole).float()
        self.color_whole_gpu      = torch.tensor(self.color_whole).float()
        self.start, self.end = [], []
        s = 0
        while s < self.uvst_whole.shape[0]:
            self.start.append(s)
            s += args.batch_size
            self.end.append(min(s, self.uvst_whole.shape[0]))
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.vis_step = 1
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999) 
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995) 
        self.epoch_num = args.whole_epoch
        # mpips eval
        self.lpips = lpips.LPIPS(net='vgg')
        if args.val is True:
            print(self.iter)
            epoch = self.iter
            self.val(epoch)
            sys.exit(0)
                 
    def train_step(self,args):
        total_train_time = datetime.timedelta()
        for epoch in range(0,self.epoch_num+1):
            if(args.loadcheckpoints):
                epoch += self.iter
            start_time = datetime.datetime.now()
            self.losses = AverageMeter()
            self.losses_rgb = AverageMeter()
            self.losses_rgb_super = AverageMeter()
            self.losses_depth = AverageMeter()
            self.model.train()
            self.step_count +=1
            perm = torch.randperm(self.uvst_whole.shape[0])
            self.uvst_whole_gpu = self.uvst_whole_gpu[perm]
            self.color_whole_gpu = self.color_whole_gpu[perm]
            self.train_loader = [{'input': self.uvst_whole_gpu[s:e], 
                                    'color': self.color_whole_gpu[s:e]} for s, e in zip(self.start, self.end)]
            pbar = tqdm(self.train_loader, desc="Train")
            for i, data_batch in enumerate(pbar):
                self.optimizer.zero_grad()
                inputs  = data_batch["input"].cuda()
                color = data_batch["color"].cuda()
                preds_color = self.model(inputs.cuda())
                loss_rgb = 1000*torch.mean((preds_color - color) * (preds_color - color))
                loss = loss_rgb 
                self.losses.update(loss.item(), inputs.size(0))
                self.losses_rgb.update(loss_rgb.item(),inputs.size(0))
                loss.backward()
                self.optimizer.step()
                log_str = 'epoch {}/{}, {}/{}, lr:{}, loss:{:4f}'.format(
                    epoch,self.epoch_num,i+1,len(self.start),
                    self.optimizer.param_groups[0]['lr'],self.losses.avg,)
                pbar.set_postfix_str(log_str)
                # if (i+1) % 2000 == 0:
                #     logging.info(log_str)
            logging.info(log_str)
            self.scheduler.step()
            end_time = datetime.datetime.now()
            epoch_time = end_time - start_time
            total_train_time += epoch_time
            logging.info(f"Training time for epoch {epoch}: {epoch_time}")
            if epoch == self.epoch_num:
                logging.info(f"Total training time: {total_train_time}")
            with torch.no_grad():
                self.model.eval()
                if epoch % args.test_freq ==0:
                    self.val(epoch)
                if epoch % args.save_checkpoints == 0:
                    cpt_path = self.checkpoints + f"nelf-{epoch}.pth"
                    torch.save(self.model.state_dict(), cpt_path)
                    print(f"Saved weights to {cpt_path}")
                #elif epoch == 250:
                #    self.val(epoch)
                #    cpt_path = self.checkpoints + f"nelf-250.pth"
                #    torch.save(self.model.state_dict(), cpt_path)
                #    print(f"Saved weights to {cpt_path}")       

    def get_uvst(self,cam_x,cam_y,cam_z, center_flag=False):
        t = np.asarray([cam_x,cam_y,cam_z])
        center = np.array([0,0,-self.fdepth])
        # get rotation matrix
        if(center_flag):
            cur_p = normalize(center-t)
            cur_o = np.array([0,0,-1])
            rot_mat = rotation_matrix_from_vectors(cur_o,cur_p)
        else:
            rot_mat = np.eye(3,dtype=float)
        c2w = np.concatenate((rot_mat,np.expand_dims(t,1)),1)
        ray_o,ray_d = get_rays(self.h,self.w,self.intrinsic,c2w)
        ray_o = np.reshape(ray_o,(-1,3))
        ray_d = np.reshape(ray_d,(-1,3))
        plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)
        # interset radius plane
        p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
        p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))
        # interset radius plane 
        inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)
        inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)
        data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)
        data_uvst /= self.norm_fac
        data_uvst = (data_uvst - self.uvst_min)/(self.uvst_max-self.uvst_min) * 2 -1.0
        return data_uvst           

    def val(self,epoch):
        with torch.no_grad():
            i=0
            p = []
            s = []
            l = []
            save_dir = self.img_folder_test + f"epoch-{epoch}"
            rm_folder(save_dir)
            total_images = self.uvst_whole_val.shape[0] // (self.w * self.h)
            with tqdm(total=total_images, desc="Validation") as pbar:
                for i in range(0, self.uvst_whole_val.shape[0], self.w * self.h):
                    end = min(i + self.w * self.h, self.uvst_whole_val.shape[0])
                    uvst = self.uvst_whole_val[i:end]
                    uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                    pred_color = self.model(uvst)
                    gt_color = self.color_whole_val[i:end]
                    pred_img = pred_color.reshape((self.h,self.w,3)).permute((2,0,1))
                    gt_img   = torch.tensor(gt_color).reshape((self.h,self.w,3)).permute((2,0,1))
                    torchvision.utils.save_image(pred_img,f"{save_dir}/test_{i}.png")
                    torchvision.utils.save_image(gt_img,f"{save_dir}/gt_{i}.png")
                    pred_color = pred_color.cpu().numpy()
                    psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
                    ssim = structural_similarity(gt_color.reshape((self.h,self.w,3)), pred_color.reshape((self.h,self.w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
                    lsp  = self.lpips(pred_img.cpu(),gt_img) 
                    p.append(psnr)
                    s.append(ssim)
                    l.append(lsp.item())
                    pbar.update(1)
            p_mean = np.asarray(p).mean()
            s_mean = np.asarray(s).mean()
            l_mean = np.asarray(l).mean()
            logging.info(f'>>> val: psnr  mean {p_mean} full {p}')
            logging.info(f'>>> val: ssim  mean {s_mean} full {s}')
            logging.info(f'>>> val: lpips mean {l_mean} full {l}')
            self.psnr_file_path = 'result/' + self.exp + '/psnr_values.txt'
            with open(self.psnr_file_path, 'a') as psnr_file:
                psnr_file.write(f'{epoch} {p_mean} {p}\n')
            # self.save_val_to_excel(epoch, [p_mean, s_mean, l_mean], [p, s, l])
            #if epoch == self.epoch_num:
            #    print(f"Saved images to {save_dir}")
            #else:
            #    shutil.rmtree(save_dir)
            # return p, s, l
            
    def save_val_to_excel(self, epoch, means, values):
        excel_file_path = f'result/{self.exp}/{self.exn}_val.xlsx'
        if not os.path.exists(excel_file_path): mode = 'w'
        else: mode = 'a'
        with pd.ExcelWriter(excel_file_path, mode=mode, engine='openpyxl', if_sheet_exists='replace') as writer:
            for i, (mean, value) in enumerate(zip(means, values)):
                sheet_name = ["PSNR", "SSIM", "LPIPS"][i]
                data = {'Epoch': [epoch], 'Loss': [self.losses.avg], 'Mean': [mean]}
                for j, val in enumerate(value):
                    data[f'{sheet_name}_{j+1}'] = val
                df = pd.DataFrame(data)
                
                if sheet_name in writer.book.sheetnames:
                    df.to_excel(writer, index=False, header=False, sheet_name=sheet_name, startrow=writer.sheets[sheet_name].max_row)
                else:
                    df.to_excel(writer, index=False, sheet_name=sheet_name)

    def train_summaries(self):
        self.summary_writer.add_scalar('total loss', self.losses.avg, self.step_count)

    def load_exp_ckpt(self, loadckptnum):
        if loadckptnum == 0:
            ckpt_paths = glob.glob(self.loadckpt+"*.pth")
            self.iter=0
            if len(ckpt_paths) > 0:
                for ckpt_path in ckpt_paths:
                    print(ckpt_path)
                    ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                    self.iter = max(self.iter, ckpt_id)
                ckpt_name = f"./{self.loadckpt}/nelf-{self.iter}.pth"
            # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        else:
            self.iter = loadckptnum
            ckpt_name = f"./{self.loadckpt}/nelf-{loadckptnum}.pth"
        print(f"Load weights from {ckpt_name}") 
        ckpt = torch.load(ckpt_name)
        try:
            self.model.load_state_dict(ckpt, strict=False)
        except:
            tmp = DataParallel(self.model)
            tmp.load_state_dict(ckpt)
            self.model.load_state_dict(tmp.module.state_dict())
            # self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            del tmp
        
    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        self.iter=0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                print(ckpt_path)
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"./{self.checkpoints}/nelf-{self.iter}.pth"
        # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        print(f"Load weights from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
        try:
            self.model.load_state_dict(ckpt)
        except:
            tmp = DataParallel(self.model)
            tmp.load_state_dict(ckpt)
            self.model.load_state_dict(tmp.module.state_dict())
            del tmp
    
if __name__ == '__main__':
    args = parser.parse_args()
    m_train = train(args)
    m_train.train_step(args)
