from pathlib import Path
from PIL import Image
from io import BytesIO
from utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
from tqdm import tqdm
from db_ops import query_db
from s3_ops import s3_client, upload_to_s3
from outfit_list import remove_invalid_outfits

import sys
import argparse
import csv
import requests

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="/home/xarchakosi/model.png", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="dc", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=1, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type
# 0:upperbody; 1:lowerbody; 2:dress
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")


def generate_image(cloth_img, model_img, masked_vton_img, mask, category):
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )
    return images


def get_image_file(image_url):
    image_response = requests.get(image_url, timeout=20)
    image_response.raise_for_status()
    return BytesIO(image_response.content)


def main():

    data = remove_invalid_outfits()
    print(data)
    exit()

    s3 = s3_client()

    with open(f"csv_files/vton.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["product_Id", "color_id", "image_url", "laydown_image_url", "seed", "model_version"])

        for index, data_row in tqdm(data.iterrows(), total=data.shape[0]):
            product_id, color_id, image_url = data_row["SF_PRODUCT_ID"], data_row["COLOR_ID"], data_row["IMAGE_URL"]

            cloth_img = Image.open(get_image_file(image_url)).resize((768, 1024)).convert("RGB")
            model_img = Image.open(model_path).resize((768, 1024)).convert("RGB")
            keypoints = openpose_model(model_img.resize((384, 512)))
            model_parse, _ = parsing_model(model_img.resize((384, 512)))
            mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
            mask = mask.resize((768, 1024), Image.NEAREST)
            mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
            masked_vton_img = Image.composite(mask_gray, model_img, mask)

            image = generate_image(cloth_img, model_img, masked_vton_img, mask)
            print(image)

            image_object = BytesIO()
            image.save(image_object, format='PNG')
            image_object.seek(0)

            output_name = f'{image_url.split("/")[-1][:-4]}_{str(seed)}.png'
            laydown_url = upload_to_s3(output_name, image_object, s3)
            writer.writerow([str(product_id), str(color_id), image_url, laydown_url])


if __name__ == '__main__':
    main()


