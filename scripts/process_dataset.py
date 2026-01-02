import os
import hashlib
import torch

from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm


def create_unique_dataset(
    original_data_path: str, 
    new_data_path: str, 
    downsampling: bool = False, 
    convert_to_png: bool = False
) -> None:
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    processed_hashes = set()

    image_files = [f for f in os.listdir(original_data_path)]

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(original_data_path, img_file)
        img: torch.Tensor = torch.load(img_path)[0:3, ...]  # (3, 128, 128)

        non_vascular_area_color = torch.tensor([-1., 1., -1.])
        vascular_area_mask = torch.any(img != non_vascular_area_color.unsqueeze(1).unsqueeze(2), dim=0)
        img[:, vascular_area_mask] = torch.tensor([255., 255., 255.]).view(3, 1)    # vascular area: white
        img[:, ~vascular_area_mask] = torch.tensor([0., 255., 0.]).view(3, 1)       # non-vascular area: green

        if downsampling:
            img = img.unsqueeze(0).float()  # Add batch dimension and convert to float type
            img = F.interpolate(img, size=(32, 32), mode='nearest')
            img = img.squeeze(0).byte()  # Remove batch dimension and convert back to uint8 type

        # Compute the hash of the image
        img_hash = hashlib.md5(img.numpy().tobytes()).hexdigest()

        # If the hash is new, save the image
        if img_hash not in processed_hashes:
            processed_hashes.add(img_hash)
            if convert_to_png:
                new_img_path = os.path.join(new_data_path, f"{img_hash}.png")
                pil_img = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')
                pil_img.save(new_img_path, format='PNG')
            else:
                new_img_path = os.path.join(new_data_path, f"{img_hash}")
                torch.save(img, new_img_path)

    print(f"There are {len(image_files)} images in the original dataset")
    print(f"Number of unique processed images: {len(processed_hashes)}")


if __name__ == '__main__':
    original_data_path = './dataset/simulated_vessel_2d'
    new_data_path = './dataset/unique_simulated_vessel_2d'
    create_unique_dataset(original_data_path, new_data_path, downsampling=False, convert_to_png=False)