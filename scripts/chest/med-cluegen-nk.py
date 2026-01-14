import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import scipy.io as sio

def make_batch(image, clue, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    clue = sio.loadmat(clue)
    clue = np.expand_dims(clue['tokens'][:,0:8],0)

    clue = torch.from_numpy(clue)



    batch = {"image": image, "caption": clue}
    #for k in batch:
    #    batch[k] = batch[k].to(device=device)
     #   batch[k] = batch[k]*2.0-1.0
    batch["image"] = batch["image"].to(device=device)
    batch["image"] = batch["image"] * 2.0 - 1.0
    batch["caption"] = batch["caption"].to(device=device)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
        default=''
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=''
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    images = glob.glob(os.path.join('', "*.png"))
    clues = glob.glob(os.path.join('', "*.mat"))
    print(f"Found {len(images)} inputs.")

    config = OmegaConf.load("D:\latent-diffusion-main\logs\chest-cluenk-base8\configs/2025-05-14T10-45-27-project.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("D:\latent-diffusion-main\logs\chest-cluenk-base8/checkpoints/last.ckpt")["state_dict"],
                          strict=False)#2024-12-09T10-28-43_c

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, clue in tqdm(zip(images, clues)):#, clue, clues
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, clue, device=device)#

                # encode masked image and concat downsampled mask
                z = batch["caption"]
                #c = model.cond_stage_model.encode(batch["masked_image"])
                #cc = torch.nn.functional.interpolate(batch["mask"],
                #                                     size=c.shape[-2:])
                #c = torch.cat((c, cc), dim=1)
                xt = model.first_stage_model.encode(batch["image"])
                xt = xt.mean
                #shape = (c.shape[1]-1,)+c.shape[2:]
                shape = (xt.shape[1] - 0,) + xt.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=z,
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
