import os
import numpy as np
import argparse
import datetime
import json
import time

import torch
from data_loaders import get_loader
from logger import Logger
from calculate_metrics import PSNR, SSIM, PerceptualLoss

from monai.networks.nets import BasicUNet, UNet, AttentionUnet

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, dataset, criterion, opt, epoch, device):
    
    model.train()
    data = iter(dataset)
    metrics = {}

    for it in range(len(dataset)):
        imgs, trgts = next(data)

        all_a, all_b = [], []
        for idx in range(imgs.size(dim=0)):
            img, trgt = imgs[idx, :, :, :], trgts[idx, :, :, :]
            for idy in range(img.size(dim=1)):
                a, b = img[:, idy, :, :], trgt[:, idy, :, :] 
                all_a.append(a)
                all_b.append(b)
        
        x, y = torch.stack(all_a), torch.stack(all_b)
        # print(x.shape, y.shape)

        x = x.to(device)
        y = y.to(device)
        y_ = model(x)

        closs = criterion(y_, y)

        opt.zero_grad()
        closs.backward()
        opt.step()

        if "loss" in metrics:
            metrics["loss"].append(closs.item())
        else:
            metrics["loss"] = [closs.item()]

        log = f"Epoch {epoch+1}, Iter {it+1}/{len(dataset)}:"
        for m in sorted(metrics.keys()):
            log += f" {m} = {np.mean(metrics[m])}"

        print(log)

    return metrics

def val_test(foldnum, totalepochs, modelname, model, dataset, criterion, criterion_mse, epoch, device, mode="val"):
    
    model.eval()
    data = iter(dataset)
    metrics = {}
    ssim_loss = SSIM(window_size=16)

    with torch.no_grad():
        for it in range(len(dataset)):
            imgs, trgts = next(data)

            all_a, all_b = [], []
            for idx in range(imgs.size(dim=0)):
                img, trgt = imgs[idx, :, :, :], trgts[idx, :, :, :]
                for idy in range(img.size(dim=1)):
                    a, b = img[:, idy, :, :], trgt[:, idy, :, :] 
                    all_a.append(a)
                    all_b.append(b)
            
            x, y = torch.stack(all_a), torch.stack(all_b)

            x = x.to(device)
            y = y.to(device)

            y_ = model(x)
            closs = criterion(y_, y)
            # content_loss, style_loss = criterion(y_, y)
            # print(content_loss, style_loss)

            data_range = (torch.max(y) - torch.min(y)).item()
            psnr_val = PSNR(y, y_, criterion_mse, data_range).item()
            ssim = ssim_loss(y, y_).item()

            if "loss" in metrics:
                metrics["loss"].append(closs.item())
            else:
                metrics["loss"] = [closs.item()]

            if "psnr" in metrics:
                metrics["psnr"].append(psnr_val)
            else:
                metrics["psnr"] = [psnr_val]

            if "ssim" in metrics:
                metrics["ssim"].append(ssim)
            else:
                metrics["ssim"] = [ssim]

        log = f"Testing on {mode.upper()} Dataset at Epoch {epoch+1}:"
        log += f" loss = {np.mean(metrics['loss'])}"
        log += f" PSNR = {np.mean(metrics['psnr'])}"
        log += f" SSIM = {np.mean(metrics['ssim'])}"

        print(log)

    return metrics

def main():

    parser = argparse.ArgumentParser(description='RIED 2D translation')
    parser.add_argument('-d', '--dataset', default='fold1', type=str)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=3, type=int, metavar='N',
                        help='number of samples in each batch')
    parser.add_argument('--model_name', default="unet", type=str,
                        help='name of the classification model')
    parser.add_argument('--num_classes', default=1, type=int, metavar='N',
                        help='number of classes to predict')
    args = parser.parse_args()


    timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    training_folder = f"/data/amciilab/jay/2dtranslation/{args.dataset}_{args.model_name}_{timestamp}"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
        os.makedirs(f"{training_folder}/weights")
    else:
        print(f"{training_folder} exists!")
        exit()

    with open(f'{training_folder}/params.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    logger = Logger(f'{training_folder}/logs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_name == "unet":
        model = BasicUNet(
            spatial_dims=2, 
            features=(16, 32, 64, 128, 256, 32),
            in_channels=1,
            out_channels=1,
        ).to(device)
    elif args.model_name == "resunet": 
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=4,
        ).to(device)
    elif args.model_name == "attunet": 
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        ).to(device)
    else:
        print(f"Invalid model name: {args.model_name}!")
        exit()

    print(args)
    print("Model param:", get_n_params(model))

    print("Working on ", args.dataset)

    dataset_train = get_loader(f'./{args.dataset}/train', batch_size=args.batch_size, mode="train")
    dataset_val = get_loader(f'./{args.dataset}/val', batch_size=1, mode="val")

    torch.backends.cudnn.benchmark = True

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = torch.nn.L1Loss()
    criterion_mse = torch.nn.MSELoss()

    for e in range(args.epochs):
        
        loss_metrics = train(model, dataset_train, criterion, opt, e, device)

        for tag, value in loss_metrics.items():
            for cnt in range(len(value)):
                logger.scalar_summary("train/"+tag, value[cnt], e * len(loss_metrics["loss"]) + cnt + 1)

        mpath = os.path.join(f"{training_folder}/weights", '{}.ckpt'.format(e+1))
        torch.save(model.state_dict(), mpath)
        
        val_loss_metrics = val_test(args.dataset, args.epochs, args.model_name, model, dataset_val, criterion, criterion_mse, e, device)
        
        for tag, value in val_loss_metrics.items():
            logger.scalar_summary("val/"+tag, np.mean(value), (e+1))


if __name__ == '__main__':
    main()


