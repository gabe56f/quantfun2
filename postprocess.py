from src.misc.postprocessors import PixelOE
from PIL import Image


def do(img: str):
    postprocessor = PixelOE()
    postprocessor.target_size = 256
    postprocessor.patch_size = 16
    postprocessor.thickness = 4
    postprocessor.colors = 32
    postprocessor.color_quant_method = "kmeans"
    postprocessor.colors_with_weight = True
    images = [Image.open(img)]
    images = postprocessor(images, 1024, 1024)
    images[0].save(f"{img}-pixel2.png")


do("arpad.png")
