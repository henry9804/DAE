# The file optimize E for embedding real-img to latent space and getting Wy.
# Please refer to 186 below to set args.
# Code-line: 205-206 for first step regularization parameters:  /beta and Norm_p

import os
import math
import torch
import lpips
import argparse
import collections
import torchvision
import numpy as np
import tensorboardX
from collections import OrderedDict
import metric.pytorch_ssim as pytorch_ssim
from training_utils import imgPath2loader, space_loss
from model.biggan_generator import BigGAN  # BigGAN
from training_utils import *
from model.utils.biggan_config import BigGANConfig
import model.E.E_BIG as BE_BIG
from model.utils.custom_adam import LREQAdam
from images.imagenet_dataset import ImgaeNetDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

# from metric.grad_cam import GradCAM, GradCamPlusPlus, GuidedBackPropagation, mask2cam
import torch.nn as nn
from tqdm import tqdm


def train(tensor_writer=None, args=None, dataloader=None):
    beta = args.beta
    rho = args.norm_p

    model_path = "./checkpoint/biggan/256/G-256.pt"
    config_file = "./checkpoint/biggan/256/biggan-deep-256-config.json"
    config = BigGANConfig.from_json_file(config_file)
    generator = BigGAN(config).to(device)
    generator.load_state_dict(torch.load(model_path), strict=False)

    batch_size = args.batch_size
    it_d = 0

    # optimize clf
    if config.clf["on"]:
        clf_optimizer = AdamW(
            [
                {"params": generator.generator.latent_clf.parameters()},
            ],
            lr=args.lr,
            betas=(args.beta_1, 0.99),
            weight_decay=0.0,
        )  # 0.0003

    latents = torch.load(resultPath1_2+'/w_all_1.pt')
    labels = torch.load(resultPath1_2+'/label_all_1.pt')

    truncation = torch.tensor(0.4, dtype=torch.float).to(device)
    for iteration in tqdm(range(0, args.iterations), desc="iteration", position=1):
        for g, (w1, label) in tqdm(enumerate(zip(latents, labels)), desc="batch", position=0):
            w1 = w1.unsqueeze(0).to(device)
            conditions = one_hot(label).unsqueeze(0).to(device)
            z = truncated_noise_sample(
                truncation=0.4, batch_size=args.batch_size, seed=args.iterations % 30000
            )
            z = torch.tensor(z, dtype=torch.float).to(device)

            loss_msiv_min = 0
            
            if config.clf["on"]:
                batch_real = []
                batch_w = []
                batch_real.append(torch.ones(args.batch_size, dtype=float, device=w1.device))
                batch_w.append(w1)
                batch_real.append(torch.zeros(args.batch_size, dtype=float, device=w1.device))
                batch_w.append(z)
                is_real = torch.concat(batch_real).reshape(-1,1)
                w_ = torch.concat(batch_w)

            # Getting
            imgs2, _ = generator(w_, conditions.repeat(2,1), truncation)
            if config.clf["on"]:
                imgs2, clf_out = imgs2

            # Classifiers
            if config.clf["on"]:
                clf_loss = 0
                for clf in clf_out:
                    # print(clf.detach().sigmoid(), is_real)
                    clf_loss += F.binary_cross_entropy_with_logits(clf, is_real)
                print(f"clf_loss={clf_loss}")
                clf_loss.backward()
                clf_optimizer.step()
                clf_optimizer.zero_grad()
            it_d += 1
            if iteration % 100 == 0:
                n_row = batch_size
                # test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
                # torchvision.utils.save_image(test_img, resultPath1_1+'/id%d_ep%d-norm%.2f.jpg'%(g,iteration,w1.norm()),nrow=2) # nrow=3
                with open(resultPath + "/Loss.txt", "a+") as f:
                    print("id_" + str(0) + "_____i_" + str(iteration), file=f)
                    print(
                        "[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]",
                        file=f,
                    )
                    print("---------ImageSpace--------", file=f)
                    # print('loss_small_info: %s'%loss_mask_info,file=f)
                    # print('loss_medium_info: %s'%loss_Gcam_info,file=f)
                    # print("loss_imgs_info: %s" % loss_imgs_info, file=f)
                    # print("loss_imgs_info: %s" % loss_imgs_info, file=f)
                    print("---------LatentSpace--------", file=f)
                    # NOTE: commented
                    # print("loss_w_info: %s" % loss_w_info, file=f)
                    # print('loss_c1_info: %s'%loss_c1_info,file=f)
                    # NOTE: commented
                    # print("loss_c2_info: %s" % loss_c2_info, file=f)
                    print("Img_loss: %s" % loss_msiv_min, file=f)

                for i, j in enumerate(w1):
                    torch.save(
                        j.unsqueeze(0),
                        resultPath1_2
                        + "/id0-i%d-w%d-norm%f.pt" % (i, iteration, w1.norm()),
                    )
                # for i,j in enumerate(imgs2):
                #     torch.save(j.unsqueeze(0),resultPath1_2+'/id%d-i%d-img%d.pt'%(g,i,iteration))
                # torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%iteration)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="the training args")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument(
        "--lr", type=float, default=0.0003
    )  # better than 0.01 W:0.003, E:0.0003
    parser.add_argument("--beta_1", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--experiment_dir", default=None)  # None
    parser.add_argument(
        "--img_dir", default="images/ILSVRC2012_img_val"
    )  # pt or directory
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--img_channels", type=int, default=3)  # RGB:3 ,L:1
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument(
        "--start_features", type=int, default=64
    )  # 16->1024 32->512 64->256
    parser.add_argument(
        "--optimizeE", type=bool, default=True
    )  # if not, optimize W directly True False
    parser.add_argument("--beta", type=float, default=10e-7)
    parser.add_argument("--norm_p", type=int, default=2)
    parser.add_argument(
        "--checkpoint_dir_E", default="./checkpoint/E/E_biggan_256_ep5.pth"
    )
    args = parser.parse_args()

    result_path = "./result_bigGAN_id30_GradCAM"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = result_path + "/mis_aligh_bigGAN_v1_opE/"
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)

    resultPath1_1 = resultPath + "/imgs"
    if not os.path.exists(resultPath1_1):
        os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath + "/models"
    if not os.path.exists(resultPath1_2):
        os.mkdir(resultPath1_2)

    writer_path = os.path.join(resultPath, "./summaries")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    img_dataset = ImgaeNetDataset(args.img_dir, (args.img_size, args.img_size))
    img_dataloader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=True)

    train(tensor_writer=writer, args=args, dataloader=img_dataloader)