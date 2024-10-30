import torch 
from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A

RESIZE = 256

class OxfordPetsSegmentation(VisionDataset):
    def __init__(self, root, txt_file, pre_image_transforms=None, pre_mask_transforms=None, both_transforms=None, post_image_transforms=None, post_mask_transforms=None):
        super(OxfordPetsSegmentation, self).__init__(root, transforms=None)
        self.root = Path(root)
        self.txt_file = self.root / "annotations" / txt_file

        self.pre_image_transforms = pre_image_transforms
        self.pre_mask_transforms = pre_mask_transforms
        self.both_transforms = both_transforms
        self.post_image_transforms = post_image_transforms
        self.post_mask_transforms = post_mask_transforms

        self.images_paths = []
        self.trimaps_paths = []
        self.classes = []
        self.species = []
        self.names = []
    
        with open(self.txt_file, "r") as f:
            pbar = tqdm(f, desc="Loading dataset")
            for line in f:
                image_name, class_id, specie, breed = line.strip().split()
                self.images_paths.append(self.root / "images" / f"{image_name}.jpg")
                self.trimaps_paths.append(self.root / "annotations/trimaps" / f"{image_name}.png")
                self.classes.append(class_id)
                self.species.append(specie)
                self.names.append(image_name)
                pbar.update(1)
            pbar.close()
        

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        # convert the image to RGB because some images are grayscale and some are RGBA
        image = Image.open(self.images_paths[idx]).convert('RGB')

        # it is best to work with images for transforms because otherwise resize and all creates artifacts
        mask = Image.open(self.trimaps_paths[idx])
        
        if self.both_transforms:
            # albumentations works with numpy arrays
            image = np.array(image)
            mask = np.array(mask)

            both = self.both_transforms(image=image, mask=mask)
            
            image = Image.fromarray(both['image'])
            mask = Image.fromarray(both['mask'])

        if self.post_image_transforms:
            image = self.post_image_transforms(image)
        if self.post_mask_transforms:
            mask = self.post_mask_transforms(mask)
        
        # if image.shape[0]
        return image, mask

def get_images_size_distribution(dataset):
    sizes = []
    for i in tqdm(range(len(dataset)), desc="Getting images size distribution"):
        image, mask = dataset[i]

        sizes.extend([*image.shape[:2]])
    return sizes

def plot_images_size_distribution(data_path, text_file):
    dataset = OxfordPetsSegmentation(data_path, text_file)

    sizes = get_images_size_distribution(dataset)

    plt.hist(sizes, bins=20)
    plt.title("Images size distribution")
    plt.savefig('images_size_distribution.png')
    print(f"Images size distribution saved in images_size_distribution.png")

    # find the median size of the images
    median_size = np.median(sizes)
    print(median_size)

def target_to_imshow(target):
    target = target.to(torch.float32)/2.
    target = target.squeeze(0)
    return target

def img_to_imshow(img):
    img = img.permute(1, 2, 0)
    return img

def mask_to_target(mask):
    mask = mask*255
    mask = mask.to(torch.int)
    mask = mask - 1 
    return mask

def mask_tensor_to_mask(mask):
    # mask = mask*255
    return mask

if __name__ == "__main__":
    data_path = Path('/tmp/oxford-pet')
    text_file = "train.txt"
    

    pre_image_transforms = None

    pre_mask_transforms = None

    both_transforms = A.Compose([
        A.Resize(RESIZE, RESIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate((-30, 30), p=0.5),
    ])

    post_image_transforms = transforms.Compose([
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # try this but normalize with correct values
        transforms.ColorJitter(contrast=0.3),
        transforms.ToTensor()
    ])

    post_mask_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(mask_to_target)
    ])


    dataset = OxfordPetsSegmentation(data_path, text_file, pre_image_transforms, pre_mask_transforms, both_transforms, post_image_transforms, post_mask_transforms)

    # plot_images_size_distribution(data_path, text_file)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=32, shuffle=True)
    img_folder = Path('images_viz')
    img_folder.mkdir(exist_ok=True)
    for i, (images, target) in enumerate(dataloader):

        plt.subplot(1, 2, 1)
        plt.imshow(img_to_imshow(images[0]))
        plt.subplot(1, 2, 2)
        plt.imshow(target_to_imshow(target[0]))
        plt.show()
        path = Path(__file__).parent / img_folder / f'image_target_{i}.png'
        plt.savefig(path)
        print(f"Image saved at {path}")
        if i == 10:
            break

    for image, mask in dataset:
        if image.shape[0] != 3:
            print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
            break

    
