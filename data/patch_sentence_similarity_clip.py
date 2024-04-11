from PIL import Image
import sys
import os
import glob
import torch
import h5py

sys.path.append("/disk/nfs/gazinasvolume1/s1985335/cr-image-narrations-ssl")
device = torch.device("cuda")
out_dir = (
    "/disk/nfs/gazinasvolume1/s1985335/cr-image-narrations-ssl/data/tmp_split_images/"
)



def clip_sentence_patch_similarity(caption, image_path, im_name, model, processor):
    patch_sentence_sim = []
    split_image_embeds = []
    model = model.to(device)
    for i in range(2 * 2):
        im_split = out_dir + im_name + "_" + str(i) + ".jpg"
        image = Image.open(im_split)
        inputs = processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(device)
        outputs = model(**inputs)
        img_embeds = outputs.image_embeds
        logits_per_image = (
            outputs.logits_per_image
        )
        probs = logits_per_image.softmax(
            dim=1
        )
        patch_sentence_sim.append(probs)
        split_image_embeds.append(img_embeds)
    patch_sentence_sim = torch.stack(patch_sentence_sim).cpu().detach().squeeze(-2)
    split_image_embeds = torch.stack(split_image_embeds).cpu().detach().squeeze(-2)

    return split_image_embeds, patch_sentence_sim


