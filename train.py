import losses
import models
import json 
from tqdm import tqdm

from data_process import OxfordPetsSegmentation, mask_to_target, RESIZE

import albumentations as A

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms

from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    save_path = Path(__file__).parent / "run_long"
    save_path.mkdir(exist_ok=True, parents=True)

    data_path = Path("/tmp/oxford-pet")
    train_txt = "train.txt"
    val_txt = "val.txt"


    pre_image_transforms = None
    pre_mask_transforms = None

    both_transforms = A.Compose([
        A.Resize(RESIZE, RESIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate((-30, 30), p=0.5),
    ])

    post_image_transforms = transforms.Compose([
        transforms.ColorJitter(contrast=0.3),
        transforms.ToTensor()
    ])  

    post_mask_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(mask_to_target)
    ])

    train_dataset = OxfordPetsSegmentation(data_path, train_txt, pre_image_transforms=pre_image_transforms, pre_mask_transforms=pre_mask_transforms, both_transforms=both_transforms, post_image_transforms=post_image_transforms, post_mask_transforms=post_mask_transforms)

    val_both_transforms = A.Compose([
        A.Resize(RESIZE, RESIZE),
    ])

    val_post_image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    val_post_mask_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(mask_to_target)
    ])

    val_dataset = OxfordPetsSegmentation(data_path, val_txt, pre_image_transforms=pre_image_transforms, pre_mask_transforms=pre_mask_transforms, both_transforms=val_both_transforms, post_image_transforms=val_post_image_transforms, post_mask_transforms=val_post_mask_transforms)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.ImageSegmentation(kernel_size=3).to(device)

    criterion = losses.IoULoss(softmax=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100

    best_iou = 0.0
    train_ious = []
    val_ious = []

    for epoch in range(num_epochs):
        model.train()
        train_iou = 0.0
        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch}")
        for i, (images, target) in enumerate(train_dataloader):
            images = images.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_iou += loss.item()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
            pbar.update(1)
        pbar.close()

        train_iou /= len(train_dataloader)
        train_ious.append(train_iou)

        # save a picture of the output
        plt.subplot(1, 3, 1)
        plt.imshow(output[0].argmax(0).cpu().detach().numpy().squeeze())
        plt.subplot(1, 3, 2)
        plt.imshow(target[0].cpu().detach().numpy().squeeze())
        plt.subplot(1, 3, 3)
        plt.imshow(images[0].permute(1, 2, 0).cpu().detach().numpy())
        plt.savefig(save_path / f"train_output_{epoch}.png")


        model.eval()

        with torch.no_grad():
            iou = 0.0
            pbar = tqdm(val_dataloader, total=len(val_dataloader), desc=f"Validation Epoch {epoch}")
            for i, (images, target) in enumerate(val_dataloader):
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                iou += losses.IoUMetric(output, target, softmax=True)
                pbar.update(1)
            pbar.close()
            iou /= len(val_dataloader)
            val_ious.append(iou.item())

            # save a picture of the output
            plt.subplot(1, 3, 1)
            plt.imshow(output[0].argmax(0).cpu().detach().numpy().squeeze())
            plt.subplot(1, 3, 2)
            plt.imshow(target[0].cpu().detach().numpy().squeeze())
            plt.subplot(1, 3, 3)
            plt.imshow(images[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.savefig(save_path / f"val_output_{epoch}.png")

            print(f"Epoch: {epoch}, Train IoU: {train_iou}, Validation IoU: {iou}")
            with open("train_val_iou.txt", "w") as f:
                f.write(f"Train IoUs: {train_ious}\n")
                f.write(f"Validation IoUs: {val_ious}\n")

            if iou > best_iou:
                best_iou = iou
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Model saved as best_model.pth with IoU: {best_iou}")

                    # save the losses and accuracies
            with open(str(Path(save_path) / "train_metrics.json"), "w", encoding='utf-8') as f:
                json.dump({"iou":train_ious}, f)
            
            with open(str(Path(save_path) / "valid_metrics.json"), "w", encoding='utf-8') as f:
                json.dump({"iou": val_ious}, f)

            print(f"Metrics saved at the end of epoch {epoch}")

        torch.save(model.state_dict(), "model.pth")
        print(f"Model saved as model.pth")
