import torchvision
import torch
import tqdm
import click
import numpy as np
import tops
from ssd import utils
from tops.config import instantiate
from PIL import Image
from vizer.draw import draw_boxes
from tops.checkpointer import load_checkpoint
from pathlib import Path
import glob
import os

@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("image_dir", type=click.Path(exists=True, dir_okay=True, path_type=str))
@click.argument("output_dir", type=click.Path(dir_okay=True, path_type=str))
@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.1)
def run_demo(config_path: Path, score_threshold: float, image_dir: Path, output_dir: Path):
    cfg = utils.load_config(config_path)
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    #image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    #output_dir.mkdir(exist_ok=True, parents=True)
    if os.path.exists(output_dir) == False:
      os.mkdir(output_dir)
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)

    for i, image_path in enumerate(tqdm.tqdm(image_paths, desc="Predicting on images")):
        image_name = Path(image_path).stem
        orig_img = np.array(Image.open(image_path).convert("RGB"))
        height, width = orig_img.shape[:2]
        img = cpu_transform({"image": orig_img})["image"].unsqueeze(0)
        img = tops.to_cuda(img)
        img = gpu_transform({"image": img})["image"]
        boxes, categories, scores = model(img,score_threshold=score_threshold)[0]
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes, categories, scores = [_.cpu().numpy() for _ in [boxes, categories, scores]]
        drawn_image = draw_boxes(
            orig_img, boxes, categories-1, scores).astype(np.uint8)
        im = Image.fromarray(drawn_image)
        #output_path = output_dir.joinpath(f"{image_name}.png")
        output_path = os.path.join(output_dir, f"{image_name}.png")
        im.save(output_path)

if __name__ == '__main__':
    run_demo()