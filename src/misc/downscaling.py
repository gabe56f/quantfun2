"""
Code mostly taken from https://github.com/KohakuBlueleaf/PixelOE under the Apache 2.0 License
Modified to use PIL instead of OpenCV
"""

from itertools import product

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.cluster import KMeans

__ALL__ = [
    "match_color",
    "color_styling",
    "color_quant",
    "expansion_weight",
    "outline_expansion",
    "contrast",
    "k_centroid",
]

kernel_expansion = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.uint8)
kernel_smoothing = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)


def match_color(source: Image.Image, target: Image.Image, level=5) -> Image.Image:
    # Convert RGB to L*a*b*, and then match the std/mean
    source_lab = np.array(source.convert("LAB")).astype(np.float32) / 255
    target_lab = np.array(target.convert("LAB")).astype(np.float32) / 255
    result = (source_lab - np.mean(source_lab)) / np.std(source_lab)
    result = result * np.std(target_lab) + np.mean(target_lab)
    source = Image.fromarray(
        (result * 255).clip(0, 255).astype(np.uint8), mode="LAB"
    ).convert("RGB")

    source = np.array(source).astype(np.float32) / 255
    target = np.array(target).astype(np.float32) / 255

    # Use wavelet colorfix method to match original low frequency data at first
    source[:, :, 0] = _wavelet_colorfix(source[:, :, 0], target[:, :, 0], level=level)
    source[:, :, 1] = _wavelet_colorfix(source[:, :, 1], target[:, :, 1], level=level)
    source[:, :, 2] = _wavelet_colorfix(source[:, :, 2], target[:, :, 2], level=level)
    output = source
    return Image.fromarray((output.clip(0, 1) * 255).astype(np.uint8), mode="RGB")


def _wavelet_colorfix(inp, target, level=5):
    inp_high, _ = _wavelet_decomposition(inp, level)
    _, target_low = _wavelet_decomposition(target, level)
    output = inp_high + target_low
    return output


def _wavelet_decomposition(inp, levels):
    high_freq = np.zeros_like(inp)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = _wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def _wavelet_blur(inp, radius):
    inp_pil = Image.fromarray((inp * 255).astype(np.uint8))
    output_pil = inp_pil.filter(ImageFilter.GaussianBlur(radius))
    output = np.array(output_pil).astype(np.float32) / 255
    return output


def color_styling(inp, saturation=1.2, contrast=1.1):
    output = inp.copy()
    output = Image.fromarray(output)

    # Saturation
    converter = ImageEnhance.Color(output)
    output = converter.enhance(saturation)

    # Contrast
    converter = ImageEnhance.Contrast(output)
    output = converter.enhance(contrast)

    return np.array(output)


def color_quant(
    image: Image.Image, colors=32, weights=None, repeats=64, method="kmeans"
) -> Image.Image:
    if method == "kmeans":
        if weights is not None:
            h, w = image.size[1], image.size[0]
            pixels = []
            weights = weights / np.max(weights) * repeats

            image_array = np.array(image)
            for i in range(h):
                for j in range(w):
                    repeat_times = max(1, int(weights[i, j]))
                    pixels.extend([image_array[i, j]] * repeat_times)

            pixels = np.array(pixels, dtype=np.float32)

            kmeans = KMeans(n_clusters=colors, random_state=0, n_init="auto").fit(
                pixels
            )
            palette = kmeans.cluster_centers_

            quantized_image = np.zeros((h, w, 3), dtype=np.uint8)
            label_idx = 0
            for i in range(h):
                for j in range(w):
                    repeat_times = max(1, int(weights[i, j]))
                    quantized_image[i, j] = palette[kmeans.labels_[label_idx]]
                    label_idx += repeat_times
            return Image.fromarray(quantized_image, mode="RGB")
        else:
            h, w = image.size[1], image.size[0]
            pixels = np.array(image).reshape((-1, 3)).astype(np.float32)

            kmeans = KMeans(n_clusters=colors, random_state=0, n_init="auto").fit(
                pixels
            )
            palette = kmeans.cluster_centers_

            quantized_image = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    quantized_image[i, j] = palette[kmeans.labels_[i * w + j]]
            return Image.fromarray(quantized_image, mode="RGB")
    elif method == "maxcover":
        img_quant = image.quantize(colors, 1, kmeans=colors).convert("RGB")
        return img_quant


@torch.no_grad()
def _apply_chunk_torch(data, kernel, stride, func):
    org_shape = data.shape
    unfold_shape = org_shape

    k_shift = max(kernel - stride, 0)
    pad_pattern = (k_shift // 2, k_shift // 2 + k_shift % 2)
    data = np.pad(data, (pad_pattern, pad_pattern), "edge")

    if len(org_shape) == 2:
        data = data[None, None, ...]

    data = F.unfold(torch.tensor(data), kernel, 1, 0, stride).transpose(-1, -2)[0]
    data[..., : stride**2] = func(data)
    data = data[None, ..., : stride**2]
    data = F.fold(
        data.transpose(-1, -2),
        unfold_shape,
        stride,
        1,
        0,
        stride,
    )[0].numpy()

    if len(org_shape) < 3:
        data = data[0]
    return data


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def expansion_weight(img: Image.Image, k=8, stride=2, avg_scale=10, dist_scale=3):
    img_y = np.array(img.convert("LAB"))[:, :, 0] / 255

    avg_y = _apply_chunk_torch(
        img_y, k * 2, stride, lambda x: torch.median(x, dim=1, keepdim=True).values
    )
    max_y = _apply_chunk_torch(
        img_y, k, stride, lambda x: torch.max(x, dim=1, keepdim=True).values
    )
    min_y = _apply_chunk_torch(
        img_y, k, stride, lambda x: torch.min(x, dim=1, keepdim=True).values
    )

    bright_dist = max_y - avg_y
    dark_dist = avg_y - min_y

    weight = (avg_y - 0.5) * avg_scale
    weight = weight - (bright_dist - dark_dist) * dist_scale

    output = _sigmoid(weight)

    output = Image.fromarray(output)
    output = output.resize(
        (img.size[0] // stride, img.size[1] // stride), Image.Resampling.BILINEAR
    )
    output = output.resize((img.size[0], img.size[1]), Image.Resampling.BILINEAR)
    output = np.array(output)

    return (
        (output - np.min(output)) / (np.max(output))
        if np.max(output) != np.min(output)
        else np.zeros_like(output)
    )


def outline_expansion(
    img: Image.Image, erode=2, dilate=2, k=16, avg_scale=10, dist_scale=3
):
    weight = expansion_weight(img, k, (k // 4) * 2, avg_scale, dist_scale)[..., None]
    orig_weight = _sigmoid((weight - 0.5) * 5) * 0.25

    img_erode = img.copy()
    for _ in range(erode):
        img_erode = img_erode.filter(ImageFilter.MinFilter(3))
    img_erode = np.array(img_erode).astype(np.float32)

    img_dilate = img.copy()
    for _ in range(dilate):
        img_dilate = img_dilate.filter(ImageFilter.MaxFilter(3))

    img_dilate = np.array(img_dilate).astype(np.float32)

    output = img_erode * weight + img_dilate * (1 - weight)
    output = output * (1 - orig_weight) + np.array(img).astype(np.float32) * orig_weight
    output = output.astype(np.uint8)

    output_pil = Image.fromarray(output)

    for _ in range(erode):
        output_pil = output_pil.filter(ImageFilter.MinFilter(3))
    for _ in range(dilate * 2):
        output_pil = output_pil.filter(ImageFilter.MaxFilter(3))
    for _ in range(erode):
        output_pil = output_pil.filter(ImageFilter.MinFilter(3))

    weight = np.abs(weight * 2 - 1) * 255
    weight_pil = Image.fromarray(weight.astype(np.uint8).squeeze(), mode="L")

    for _ in range(dilate):
        weight_pil = weight_pil.filter(ImageFilter.MaxFilter(3))

    return output_pil, weight_pil


def _find_pixel(chunks):
    mid = chunks[..., chunks.shape[-1] // 2][..., None]
    med = torch.median(chunks, dim=1, keepdims=True).values
    mu = torch.mean(chunks, dim=1, keepdims=True)
    maxi = torch.max(chunks, dim=1, keepdims=True).values
    mini = torch.min(chunks, dim=1, keepdims=True).values

    output = mid
    mini_loc = (med < mu) & (maxi - med > med - mini)
    maxi_loc = (med > mu) & (maxi - med < med - mini)

    output[mini_loc] = mini[mini_loc]
    output[maxi_loc] = maxi[maxi_loc]

    return output


def contrast(
    img: Image.Image,
    target_size=128,
):
    H, W = img.size

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    patch_size = max(int(round(H // target_hw[1])), int(round(W // target_hw[0])))

    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab = np.array(img.convert("LAB")).astype(np.float32)
    img_lab[:, :, 0] = _apply_chunk_torch(
        img_lab[:, :, 0], patch_size, patch_size, _find_pixel
    )
    img_lab[:, :, 1] = _apply_chunk_torch(
        img_lab[:, :, 1],
        patch_size,
        patch_size,
        lambda x: torch.median(x, dim=1, keepdims=True).values,
    )
    img_lab[:, :, 2] = _apply_chunk_torch(
        img_lab[:, :, 2],
        patch_size,
        patch_size,
        lambda x: torch.median(x, dim=1, keepdims=True).values,
    )
    img = Image.fromarray(
        np.clip(img_lab, 0, 255).astype(np.uint8), mode="LAB"
    ).convert("RGB")
    # img = cv2.cvtColor(img_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    img_sm = img.resize(target_hw, Image.Resampling.NEAREST)
    # img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_NEAREST)
    return img_sm


def k_centroid(img: Image.Image, target_size: int = 128, centroids: int = 2):
    """
    k-centroid downscaling algorithm from Astropulse, under MIT License.
    https://github.com/Astropulse/pixeldetector/blob/6e88e18ddbd16529b5dd85b1c615cbb2e5778bf2/k-centroid.py#L19-L44
    """
    H, W = img.size

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    height = int(target_size)
    width = int(target_size * ratio)

    # Downscale outline expanded image with k-centroid
    # Create an empty array for the downscaled image
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factors
    wFactor = img.width / width
    hFactor = img.height / height

    # Iterate over each tile in the downscaled image
    for x, y in product(range(width), range(height)):
        # Crop the tile from the original image
        tile = img.crop(
            (x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor)
        )

        # Quantize the colors of the tile using k-means clustering
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert(
            "RGB"
        )

        # Get the color counts and find the most common color
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]

        # Assign the most common color to the corresponding pixel in the downscaled image
        downscaled[y, x, :] = most_common_color

    return Image.fromarray(downscaled, mode="RGB")
