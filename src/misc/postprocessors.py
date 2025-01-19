from typing import Literal

import numpy as np
from PIL import Image

from .downscaling import (
    color_quant,
    color_styling,
    expansion_weight,
    match_color,
    outline_expansion,
    k_centroid,
    contrast,
)
from ..pipeline import Postprocessor, Images, Pipelinelike


class PixelOE(Postprocessor):
    mode: Literal["contrast", "k_centroid"] = "contrast"
    target_size: int = 128
    patch_size: int = 16
    thickness: int = 2
    color_matching: bool = True
    contrast: float = 1.0
    saturation: float = 1.0
    color_quant_method: Literal["kmeans", "maxcover"] = "kmeans"
    colors: int = None
    colors_with_weight: bool = False
    no_downscale: bool = False

    def setup(self, pipeline: Pipelinelike) -> Postprocessor:
        return self

    def __call__(self, images: Images, width: int, height: int) -> Images:
        def process(img: Image.Image) -> Image.Image:

            # make sure its rgb
            if img.mode != "RGB":
                img = img.convert("RGB")

            H, W = img.size
            ratio = W / H
            target_org_size = (self.target_size**2 * self.patch_size**2 / ratio) ** 0.5
            target_org_hw = (int(target_org_size * ratio), int(target_org_size))

            img = img.resize(target_org_hw, Image.Resampling.BILINEAR)
            org_img = img.copy()

            if self.thickness is not None:
                img, weight = outline_expansion(
                    img, self.thickness, self.thickness, self.patch_size, 9, 4
                )
            elif self.colors is not None and self.colors_with_weight:
                weight = expansion_weight(
                    img, self.patch_size, (self.patch_size // 4) * 2, 9, 4
                )[..., None]
                weight = np.abs(weight * 2 - 1)

            if self.color_matching:
                img = match_color(img, org_img)

            if self.no_downscale:
                return img
            if self.mode == "contrast":
                img_sm: Image.Image = contrast(img)
            elif self.mode == "kmeans":
                # raise NotImplementedError("kmeans is not implemented yet.")
                img_sm = k_centroid(img)

            if self.colors is not None:
                img_sm_orig = img_sm.copy()
                weight_mat = None
                if self.colors_with_weight:
                    weight_mat = weight.resize(img_sm.size, Image.Resampling.BILINEAR)
                    weight_mat = np.array(weight_mat)
                    weight_gamma = self.target_size / 512
                    weight_mat = weight_mat**weight_gamma
                img_sm = color_quant(
                    img_sm,
                    self.colors,
                    weight_mat,
                    int((self.patch_size * self.colors) ** 0.5),
                )
                img_sm = match_color(img_sm, img_sm_orig, 3)

            if self.contrast != 1.0 or self.saturation != 1.0:
                img_sm = color_styling(img_sm, self.contrast, self.saturation)

            return img_sm.resize(
                (img_sm.size[0] * self.patch_size, img_sm.size[1] * self.patch_size),
                Image.Resampling.NEAREST,
            )

        return [process(img) for img in images]
