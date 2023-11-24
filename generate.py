import os
import numpy as np
import argparse
import datetime
import json
import time
import glob
import csv

import torch
from data_loaders_name import get_loader

from monai.networks.nets import BasicUNet, UNet, AttentionUnet

from monai.inferers import sliding_window_inference
import nibabel as nib

def val_test(foldnum, syn_folder, modelname, model, dataset, criterion, criterion_mse, device, mode="val"):
    
    model.eval()
    data = iter(dataset)
    metrics = {}

    paired_raw, paired = {}, {}
    with open("paired.csv", "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            paired_raw[row[0]] = row[1]
            paired[row[0].split('/')[-1].split('.nii')[0]] = row[1].split('/')[-1].split('.nii')[0]

    with torch.no_grad():
        for it in range(len(dataset)):
            imgs, trgts, name = next(data)
            # print(imgs.size(), trgts.size())

            syn_x, syn_y = [], []
            for idx in range(imgs.size(dim=0)):
                img, trgt = imgs[idx, :, :, :], trgts[idx, :, :, :]
                for idy in range(img.size(dim=1)):
                    x, y = img[:, idy, :, :], trgt[:, idy, :, :] 
                    x = x.to(device)
                    y = y.to(device)
                    x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)

                    y_ = model(x)
                    syn_x.append(y_)

            y_ = torch.cat(syn_x, dim=1)   

            y_ = y_.data.cpu().numpy()
            y_ = np.squeeze(y_)

            real_pib = paired_raw[name[0]]
            name = name[0].split("/")[-1].split(".")[0]

            real_pib_img = nib.load(real_pib)

            syn_image = nib.Nifti1Image(y_, real_pib_img.affine, real_pib_img.header)
            nib.save(syn_image, f"{syn_folder}/syn_{paired[name]}")
            
            print(f"Done generating {name}")

    return metrics

def main():

    parser = argparse.ArgumentParser(description='RIED 2D translation')
    parser.add_argument('--model_name', default="resunet", type=str,
                        help='name of the classification model')
    parser.add_argument('-d', '--dataset', default='adni', type=str)
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='number of samples in each batch')
    parser.add_argument('--num_classes', default=1, type=int, metavar='N',
                        help='number of classes to predict')
    args = parser.parse_args()

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

    mpath = f"path_to_best_model.ckpt"
    
    model.load_state_dict(torch.load(mpath, map_location=device))
    dataset_val = get_loader(f'./{args.dataset}/val', batch_size=1, mode="val")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = torch.nn.L1Loss()
    criterion_mse = torch.nn.MSELoss()

    syn_folder = "./results/"

    if not os.path.exists(syn_folder):
        os.makedirs(syn_folder)
        
    val_loss_metrics = val_test(args.dataset, syn_folder, args.model_name, model, dataset_val, criterion, criterion_mse, device)

if __name__ == '__main__':
    main()