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

    # vgg11->Grad-CAM
    # vgg11 = torchvision.models.vgg11(pretrained=True).to(device)
    # final_layer = None
    # for name, m in vgg11.named_modules():
    #     if isinstance(m, nn.Conv2d):
    #         final_layer = name
    # grad_cam_plus_plus = GradCamPlusPlus(vgg11, final_layer)
    # gbp = GuidedBackPropagation(vgg11)

    E = BE_BIG.BE(
        startf=args.start_features,
        maxf=512,
        layer_count=int(math.log(args.img_size, 2) - 1),
        latent_size=512,
        channels=3,
        biggan=True,
    ).to(device)
    if args.checkpoint_dir_E is not None:
        E.load_state_dict(
            torch.load(args.checkpoint_dir_E, map_location=torch.device(device))
        )
        if config.freeze_enc:
            for p in E.parameters():
                p.requires_grad = False
    writer = tensor_writer
    loss_lpips = lpips.LPIPS(net="vgg").to(device)
    batch_size = args.batch_size
    it_d = 0

    # optimize E
    args.optimizeE = False
    if args.optimizeE == True:
        E_optimizer = LREQAdam(
            [
                {"params": E.parameters()},
            ],
            lr=args.lr,
            betas=(args.beta_1, 0.99),
            weight_decay=0.0,
        )  # 0.0003

    # optimize clf
    if config.clf["on"]:
        clf_optimizer = AdamW(
            [
                {"params": generator.generator.latent_clf.parameters()},
            ],
            # lr=args.lr,
            # betas=(args.beta_1, 0.99),
            # weight_decay=0.0,
        )  # 0.0003

    w_all = []
    label_all = []
    img_all = []

    for g, (imgs1, labels) in tqdm(enumerate(dataloader), desc="batch", position=0):
        truncation = torch.tensor(0.4, dtype=torch.float).to(device)
        conditions = one_hot(labels).to(device)
        embed = generator.embeddings(conditions)  # 1000 => z_dim: 128
        z = truncated_noise_sample(
            truncation=0.4, batch_size=args.batch_size, seed=args.iterations % 30000
        )
        z = torch.tensor(z, dtype=torch.float).to(device)
        cond_vector = torch.cat((z, embed), dim=1)  # 128->256
        # cond_vector.requires_grad=True
        imgs1 = imgs1.to(device)
        if args.optimizeE == False:
            const1, w1_ = E(imgs1, cond_vector)
            w1 = w1_.detach()
            w1.requires_grad = True
            E_optimizer = LREQAdam(
                [
                    {"params": w1},
                ],
                lr=args.lr,
                betas=(args.beta_1, 0.99),
                weight_decay=0,
            )
        else:
            E.load_state_dict(
                torch.load(args.checkpoint_dir_E)
            )  # if not this reload, the max num of optimizing images is about 5-6.
            E_optimizer.state = collections.defaultdict(
                dict
            )  # Fresh the optimizer state. E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0)
        loss_msiv_min = 0
        for iteration in tqdm(range(0, args.iterations), desc="iteration", position=1):
            if args.optimizeE == True:
                _, w1 = E(imgs1, cond_vector)
            # imgs2 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
            imgs2, _ = generator(
                w1, conditions, truncation
            )
            _, w2 = E(imgs2, cond_vector)

            # Image
            loss_imgs, loss_imgs_info = space_loss(imgs1, imgs2, lpips_model=loss_lpips)
            E_optimizer.zero_grad()
            loss_msiv = loss_imgs  # + loss_mask + loss_Gcam
            loss_msiv.backward(retain_graph=True)  # retain_graph=True
            E_optimizer.step()

            ##Latent-Vectors
            loss_w, loss_w_info = space_loss(w1, w2, image_space=False)
            E_optimizer.zero_grad()
            loss_msLv = loss_w * 0.01
            loss_msLv.backward(retain_graph=True)  # retain_graph=True
            E_optimizer.step()
            """
            if iteration == args.iterations//2:
                loss_msiv_min = loss_msiv.item()

            if loss_msiv_min > loss_msiv.item()*1.05:
                loss_msiv_min = loss_msiv.item()
                torch.save(w1,resultPath1_2+'/id%d-iter%d-norm%f-imgLoss-min%f.pt'%(g,iteration,w1.norm(),loss_msiv_min))
                # test_img_min1 = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
                # torchvision.utils.save_image(test_img_min1, resultPath1_1+'/id%d_ep%d-norm%.2f-imgLoss-min%f.jpg'%(g, iteration, w1.norm(), loss_msiv_min),nrow=2)
                with open(resultPath+'/loss_min.txt','a+') as f:
                    print('ep%d_iter%d_minImg%.5f_wNorm%f'%(g,iteration,loss_msiv_min,w1.norm()),file=f)
            """

            it_d += 1
            if iteration % 100 == 0:
                n_row = batch_size
                # test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
                # torchvision.utils.save_image(test_img, resultPath1_1+'/id%d_ep%d-norm%.2f.jpg'%(g,iteration,w1.norm()),nrow=2) # nrow=3
                with open(resultPath + "/Loss.txt", "a+") as f:
                    print("id_" + str(g) + "_____i_" + str(iteration), file=f)
                    print(
                        "[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]",
                        file=f,
                    )
                    print("---------ImageSpace--------", file=f)
                    print("loss_imgs_info: %s" % loss_imgs_info, file=f)
                    print("---------LatentSpace--------", file=f)
                    print("loss_w_info: %s" % loss_w_info, file=f)
                    print("Img_loss: %s" % loss_msiv_min, file=f)

                for i, j in enumerate(w1):
                    torch.save(
                        j.unsqueeze(0),
                        resultPath1_2
                        + "/id%d-i%d-w%d-norm%f.pt" % (g, i, iteration, w1.norm()),
                    )
                # for i,j in enumerate(imgs2):
                #     torch.save(j.unsqueeze(0),resultPath1_2+'/id%d-i%d-img%d.pt'%(g,i,iteration))
                # torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%iteration)

        test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
        torchvision.utils.save_image(test_img, resultPath1_1+'/id%d-norm%.2f.jpg'%(g,w1.norm()),nrow=2) # nrow=3

        torchvision.utils.save_image(
            imgs2 * 0.5 + 0.5, writer_path + "/%s_rec.png" % str(g).rjust(5, "0")
        )
        w_all.append(w1[0].cpu())
        label_all.append(labels[0].cpu())
        img_all.append(imgs2[0].cpu())

    w_all_tensor = torch.stack(w_all, dim=0)
    label_all_tensor = torch.stack(label_all, dim=0)
    img_all_tensor = torch.stack(img_all, dim=0)
    torch.save(w_all_tensor, resultPath1_2 + "/w_all_%d.pt" % g)
    torch.save(label_all_tensor, resultPath1_2 + "/label_all_%d.pt" % g)
    torch.save(img_all_tensor, resultPath1_2 + "/img_all_%d.pt" % g)


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
