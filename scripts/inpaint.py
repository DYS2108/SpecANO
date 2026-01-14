import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    '''
    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    '''

    batch = {"image": image}
    batch["image"] = batch["image"].to(device=device)
    batch["image"] = batch["image"] * 2.0 - 1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
        default='D:\chest-data-origin/final'
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default='midpkuph_un'
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

   # masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    #masks = glob.glob(os.path.join('D:\chest-data-origin\pkuph\mask', "*.png"))
    #images = [x.replace("_mask.png", ".png") for x in masks]
    images = glob.glob(os.path.join('D:/chest-data-origin/final/img2d', "*.png"))
    print(f"Found {len(images)} inputs.")

    config = OmegaConf.load("D:/latent-diffusion-main/configs/latent-diffusion/med-ldm-kl-16.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("D:/latent-diffusion-main/logs/16ldm/checkpoints/last.ckpt")["state_dict"],
                          strict=False)#2024-12-09T10-28-43_c

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image in tqdm(images):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, device=device)

                # encode masked image and concat downsampled mask
                #z = model.first_stage_model.encode(batch["image"])
                #c = model.cond_stage_model.encode(batch["masked_image"])
                #cc = torch.nn.functional.interpolate(batch["mask"],
                #                                     size=c.shape[-2:])
                xt = model.first_stage_model.encode(batch["image"])
                xt = xt.mean
                # shape = (c.shape[1]-1,)+c.shape[2:]
                shape = (xt.shape[1] - 0,) + xt.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 x_T=xt,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                #image = torch.clamp((batch["image"]+1.0)/2.0,
                 #                   min=0.0, max=1.0)
                #mask = torch.clamp((batch["mask"]+1.0)/2.0,
                 #                  min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
