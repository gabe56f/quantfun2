from typing import Protocol, Optional, Tuple, List, Union, Dict
import inspect

import einops
import numpy as np
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
import torch
import torchvision.transforms as T
from tqdm.auto import tqdm
from PIL import Image

from .models import NextDiT
from .misc.image_utils import (
    get_closest_ratio,
    CenterCropResizeImage,
)
from .quant import qdtype, quantize_model


Prompt = Union[str, Tuple[str, bool]]
Prompts = List[Prompt]
Images = List[Image.Image]
Pseudorandom = Union[torch.Generator, int]
Datatype = Union[torch.dtype, qdtype]


class Schedulerlike(Protocol):
    timesteps: torch.Tensor
    init_noise_sigma: torch.Tensor

    def set_timesteps(self, num_inference_steps: int, device: torch.device): ...

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
    ) -> tuple: ...


class Pipelinelike:
    def prompt(
        self,
        prompts: Prompts,
        images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int]: ...

    def prepare_latents(
        self,
        batch_size: int,
        latent_channels: int = 4,
        height: int = 1024,
        width: int = 1024,
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        image: torch.Tensor = None,
    ) -> torch.Tensor: ...

    def to(
        self,
        device: torch.device,
        dtype: Union[Dict[str, qdtype], Union[torch.dtype, qdtype]] = torch.bfloat16,
    ): ...

    def decode_image(
        self,
        latents: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
    ) -> List[Image.Image]: ...

    def encode_image(
        self,
        image: Images,
        image_settings: "ImageSettings",
        height: int = 1024,
        width: int = 1024,
    ) -> Optional[torch.Tensor]: ...

    def prepare_extra_step_kwargs(self, **kwargs) -> Dict[str, any]:
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        extra_kwargs = {}
        for k, v in kwargs.items():
            if k in set(inspect.signature(self.scheduler.step).parameters.keys()):
                extra_kwargs[k] = v
        return extra_kwargs


def requires(load: str):
    def decorator(func):
        def inner(self, *args, **kwargs):
            if isinstance(self, Pipelinelike):
                if self.offload:
                    for name, model in self.models.items():
                        if name in load:
                            model.to(self.device)
                        else:
                            model.to("cpu")
                    torch.cuda.empty_cache()
            return func(self, *args, **kwargs)

        return inner

    return decorator


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b

    return mu


def retrieve_timesteps(
    scheduler: Schedulerlike,
    num_inference_steps: int,
    device: Optional[torch.device] = None,
    sigmas: list = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    scheduler.set_timesteps(num_inference_steps, device, sigmas=sigmas, **kwargs)

    return scheduler.timesteps, num_inference_steps


class ImageSettings:
    crop: bool

    multiview: bool
    azimuths: List[int]
    elevations: List[int]
    distances: List[float]
    c2ws: List[torch.Tensor]
    intrinsics: torch.Tensor
    focal_length: float

    def __init__(
        self,
        crop: bool = True,
        multiview: bool = False,
        azimuths: List[int] = [0, 30, 60, 90],
        elevations: List[int] = [0, 0, 0, 0],
        distances: float = 1.7,
        c2ws: List[torch.Tensor] = None,
        intrinsics: torch.Tensor = None,
        focal_length: float = 1.3887,
    ):
        self.crop = crop
        self.multiview = multiview
        self.azimuths = azimuths
        self.elevations = elevations
        if not isinstance(distances, list) and not isinstance(distances, tuple):
            self.distances = [distances] * len(azimuths)
        else:
            self.distances = distances
        self.c2ws = c2ws
        self.intrinsics = intrinsics
        self.focal_length = focal_length


DEFAULT_SETTINGS = ImageSettings()


class OneDiffusionPipeline(Pipelinelike):
    def __init__(
        self,
        transformer: NextDiT,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        scheduler: Schedulerlike,
    ) -> None:
        # super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.models = {
            "transformer": self.transformer,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
        }

        self.offload = False
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        self.max_length = 300

    def to(
        self,
        device: torch.device,
        dtype: Optional[Union[Dict[str, Datatype], Datatype]] = None,
    ):
        if dtype is None:
            self.transformer.to(device)
            self.text_encoder.to(device)
            self.vae.to(device)
        else:
            if isinstance(dtype, dict):
                if isinstance(dtype["transformer"], torch.dtype):
                    self.transformer.to(device=device, dtype=dtype["transformer"])
                else:
                    quantize_model(
                        self.transformer, dtype["transformer"](), device=device
                    )
                if isinstance(dtype["text_encoder"], torch.dtype):
                    self.text_encoder.to(device=device, dtype=dtype["text_encoder"])
                else:
                    quantize_model(
                        self.text_encoder, dtype["text_encoder"](), device=device
                    )
                self.vae.to(device, torch.bfloat16)
            elif not isinstance(dtype, torch.dtype):
                dtype = dtype()

                quantize_model(self.transformer, dtype, device=device)
                quantize_model(self.text_encoder, dtype, device=device)
                self.vae.to(device, torch.bfloat16)
            else:
                self.transformer.to(device, dtype)
                self.vae.to(device, dtype)
                self.text_encoder.to(device, dtype)

    @requires("text_encoder")
    def prompt(
        self,
        prompts: Prompts,
        images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int]:
        dt = torch.float32

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [(p, False) if isinstance(p, str) else p for p in prompts]

        positive = list(map(lambda x: x[0], filter(lambda x: not x[1], prompts)))
        negative = list(map(lambda x: x[0], filter(lambda x: x[1], prompts)))

        text_inputs = self.tokenizer(
            positive,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        text_encoder_output = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            attention_mask=attention_mask.to(self.text_encoder.device),
        )
        prompt_embeds = text_encoder_output[0].to(dt)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, images_per_prompt, 1).view(
            bs_embed * images_per_prompt, seq_len, -1
        )
        attention_mask = attention_mask.repeat(1, images_per_prompt).view(
            bs_embed * images_per_prompt, -1
        )

        if len(negative) > 0:
            uncond_text_inputs = self.tokenizer(
                negative,
                padding="max_length",
                max_length=text_input_ids.shape[-1],
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_encoder_output = self.text_encoder(
                uncond_text_inputs.input_ids.to(self.text_encoder.device),
                attention_mask=uncond_text_inputs.attention_mask.to(
                    self.text_encoder.device
                ),
            )
            uncond_prompt_embeds = uncond_text_encoder_output[0].to(dt)
            uncond_prompt_embeds = uncond_prompt_embeds.repeat(
                1, images_per_prompt, 1
            ).view(bs_embed * images_per_prompt, seq_len, -1)
            uncond_attention_mask = uncond_text_inputs.attention_mask.repeat(
                1, images_per_prompt
            ).view(bs_embed * images_per_prompt, -1)
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
            attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        return (
            prompt_embeds.to(self.text_encoder.device),
            attention_mask.to(self.text_encoder.device),
            len(negative) > 0,
            bs_embed,
        )

    @requires("transformer,vae")
    def prepare_latents(
        self,
        batch_size: int,
        latent_channels: int = 4,
        height: int = 1024,
        width: int = 1024,
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        image: torch.Tensor = None,
    ) -> torch.Tensor:
        # nchw
        shape = (
            batch_size,
            latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if generator is None:
            generator = torch.Generator(self.transformer.device)
            generator = generator.manual_seed(torch.seed())

        if latents is None:
            latents = torch.randn(
                shape,
                generator=generator,
                device=self.transformer.device,
                dtype=self.dtype,
            )
        else:
            latents = latents.to(dtype=self.dtype, device=self.transformer.device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        if image is None:
            return latents

        image = image.to(device=self.device, dtype=self.dtype)
        vae_output = self.vae.encode(image.to(self.vae.dtype))
        if hasattr(vae_output, "latent_dist"):
            init_latents: torch.Tensor = vae_output.latent_dist.sample(generator)
        else:
            init_latents = vae_output.latents
        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents = init_latents.to(self.device, dtype=self.dtype)
        init_latents = einops.rearrange(
            init_latents,
            "(bs views) c h w -> bs views c h w",
            bs=batch_size,
            views=init_latents.shape[0] // batch_size,
        )
        return init_latents

    @requires("vae")
    def decode_image(
        self, latents: torch.Tensor, height: int = 1024, width: int = 1024
    ) -> List[Image.Image]:
        image = self.vae.tiled_decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if image.ndim == 3:
            image = image[None, ...]
        image = (image * 255).round().astype("uint8")
        if image.shape[-1] == 1:
            images = [Image.fromarray(img.squeeze(), mode="L") for img in image]
        else:
            images = [Image.fromarray(img) for img in image]
        return images

    def encode_image(
        self,
        image: Images,
        image_settings: ImageSettings,
        height: int = 1024,
        width: int = 1024,
    ) -> Optional[torch.Tensor]:
        if image is None or len(image) == 0:
            return None
        if image_settings.crop:
            transforms = T.Compose(
                [
                    T.Lambda(lambda image: image.convert("RGB")),
                    T.ToTensor(),
                    CenterCropResizeImage((height, width)),
                    T.Normalize([0.5], [0.5]),
                ]
            )
        else:
            transforms = T.Compose(
                [
                    T.Lambda(lambda image: image.convert("RGB")),
                    T.ToTensor(),
                    T.Resize((height, width)),
                    T.Normalize([0.5], [0.5]),
                ]
            )

        processed_image = torch.stack([transforms(img) for img in image])
        if processed_image.min() >= 0:
            processed_image = 2.0 * processed_image - 1.0
        return processed_image

    def prepare_init_latents(
        self,
        batch_size: int,
        sequence_length: int,
        latent_channels: int = 4,
        height: int = 1024,
        width: int = 1024,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            sequence_length,
            latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        return latents

    def calculate_multiview(
        self,
        image_settings: ImageSettings,
        cond_indices: torch.Tensor,
        batch_size: int,
        images_per_prompt: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .misc.ray_utils import calculate_rays, create_c2w_matrix

        cond_indices_images = [index // 2 for index in cond_indices if index % 2 == 0]
        cond_indices_rays = [index // 2 for index in cond_indices if index % 2 == 1]

        elevations = [
            element for element in image_settings.elevations if element is not None
        ]
        azimuths = [
            element for element in image_settings.azimuths if element is not None
        ]
        distances = [
            element for element in image_settings.distances if element is not None
        ]
        if image_settings.c2ws is None:
            c2ws = [
                torch.tensor(
                    create_c2w_matrix(azimuth, elevation, distance)
                    for azimuth, elevation, distance in zip(
                        azimuths, elevations, distances
                    )
                )
            ]
            c2ws = torch.stack(c2ws).float()
        else:
            c2ws = torch.tensor(c2ws).float()

        c2ws[:, 0:3, 1:3] *= -1
        c2ws = c2ws[:, [1, 0, 2, 3], :]
        c2ws[:, 2, :] *= -1

        w2cs = torch.inverse(c2ws)
        if image_settings.intrinsics is None:
            K = torch.tensor(
                [
                    [
                        [image_settings.focal_length, 0, 0.5],
                        [0, image_settings.focal_length, 0.5],
                        [0, 0, 1],
                    ]
                ]
            ).repeat(c2ws.shape[0], 1, 1)
        else:
            K = image_settings.intrinsics
        Rs = w2cs[:, :3, :3]
        Ts = w2cs[:, :3, 3]
        sizes = torch.tensor([[1, 1]]).repeat(c2ws.shape[0], 1)

        cond_rays = calculate_rays(K, sizes, Rs, Ts, height // 8)
        cond_rays = cond_rays.reshape(-1, height // 8, width // 8, 6)
        cond_rays = (
            torch.cat([cond_rays, cond_rays, cond_rays[..., :4]], dim=-1) * 1.658
        )
        cond_rays[None].repeat(batch_size * images_per_prompt, 1, 1, 1, 1)
        cond_rays = cond_rays.permute(0, 1, 4, 2, 3)
        cond_rays = cond_rays.to(self.device, dtype=self.dtype)

        return cond_indices_images, cond_indices_rays, cond_rays

    @torch.no_grad()
    def __call__(
        self,
        prompts: Prompts,
        image: Images = None,
        image_settings: ImageSettings = DEFAULT_SETTINGS,
        size: Tuple[int, int] = None,
        images_per_prompt: int = 1,
        steps: int = 24,
        cfg: float = 3.5,
        seed: Pseudorandom = 1337,
        eta: float = 0.0,
        denoise_mask: List[int] = [1, 0],
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        if isinstance(seed, int):
            generator = torch.Generator(self.device)
            generator.manual_seed(seed)
        else:
            generator = seed

        if image is not None and not isinstance(image, list):
            image: Images = [image]

        if size is None and image is not None:
            closest_ar = get_closest_ratio(image[0].size[1], image[0].size[0])
            size = (int(closest_ar[0][0]), int(closest_ar[0][1]))
        elif size is None:
            size = (1024, 1024)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator=generator, eta=eta)
        encoder_hidden_states, encoder_attention_mask, do_cfg, batch_size = self.prompt(
            prompts, images_per_prompt
        )

        processed_image = self.encode_image(image, image_settings, *size)

        latents = self.prepare_latents(
            batch_size * images_per_prompt,
            self.transformer.config.in_channels,
            *size,
            generator=generator,
            image=processed_image,
        )
        cond_latents: torch.Tensor = None
        denoise_indices: torch.Tensor = None
        if processed_image is not None:
            denoise_mask = torch.tensor(denoise_mask, device=self.device)
            denoise_indices = torch.where(denoise_mask == 1)[0]
            cond_indices = torch.where(denoise_mask == 0)[0]
            sequence_length = denoise_mask.shape[0]
            cond_latents = latents
            latents = self.prepare_init_latents(
                batch_size * images_per_prompt,
                sequence_length,
                self.transformer.config.in_channels,
                *size,
                generator=generator,
            )

            image_seq_len = (
                noise_scale
                * sum(denoise_mask)
                * latents.shape[-1]
                * latents.shape[-2]
                / self.transformer.config.patch_size[-1]
                / self.transformer.config.patch_size[-2]
            )
        else:
            image_seq_len = (
                latents.shape[-1]
                * latents.shape[-2]
                / self.transformer.config.patch_size[-1]
                / self.transformer.config.patch_size[-2]
            )
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, steps = retrieve_timesteps(
            self.scheduler, steps, self.device, sigmas=sigmas, mu=mu
        )

        if image_settings.multiview:
            cond_indices_images, cond_indices_rays, cond_rays = (
                self.calculate_multiview(
                    image_settings, cond_indices, batch_size, images_per_prompt, *size
                )
            )
            latents = einops.rearrange(latents, "b (f n) c h w -> b f n c h w", n=2)
            if cond_latents is not None:
                latents[:, cond_indices_images, 0] = cond_latents
            latents[:, cond_indices_rays, 1] = cond_rays
            latents = einops.rearrange(latents, "b f n c h w -> b (f n) c h w")
        else:
            if cond_latents is not None:
                latents[:, cond_indices] = cond_latents

        for i, t in tqdm(enumerate(timesteps), total=steps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            if cond_latents is None and not image_settings.multiview:
                timestep = torch.tensor(
                    [t] * latent_model_input.shape[0],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                timestep = torch.broadcast_to(
                    einops.repeat(
                        torch.tensor([t], device=self.device, dtype=torch.float32),
                        "1 -> 1 f 1 1 1",
                        f=latent_model_input.shape[1],
                    ),
                    latent_model_input.shape,
                ).clone()

                if image_settings.multiview:
                    timestep = einops.rearrange(
                        timestep, "b (f n) c h w -> b f n c h w", n=2
                    )
                    timestep[:, cond_indices_images, 0] = self.scheduler.timesteps[-1]
                    timestep[:, cond_indices_rays, 1] = self.scheduler.timesteps[-1]
                    timestep = einops.rearrange(
                        timestep, "b f n c h w -> b (f n) c h w"
                    )
                else:
                    timestep[:, cond_indices] = self.scheduler.timesteps[-1]
            noise_pred = self.transformer(
                samples=latent_model_input.to(self.dtype),
                timesteps=timestep,
                encoder_hidden_states=encoder_hidden_states.to(self.dtype),
                encoder_attention_mask=encoder_attention_mask,
            )

            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg * (
                    noise_pred_cond - noise_pred_uncond
                )

            if cond_latents is None and not image_settings.multiview:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
            else:
                bs, n_frame = noise_pred.shape[:2]
                noise_pred = einops.rearrange(noise_pred, "b f c h w -> (b f) c h w")
                latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                latents = einops.rearrange(
                    latents, "(b f) c h w -> b f c h w", b=bs, f=n_frame
                )
                if image_settings.multiview:
                    latents = einops.rearrange(
                        latents, "b (f n) c h w -> b f n c h w", n=2
                    )
                    if cond_latents is not None:
                        latents[:, cond_indices_images, 0] = cond_latents
                    latents[:, cond_indices_rays, 1] = cond_rays
                    latents = einops.rearrange(latents, "b f n c h w -> b (f n) c h w")
                else:
                    if cond_latents is not None:
                        latents[:, cond_indices] = cond_latents

        latents = 1 / self.vae.config.scaling_factor * latents
        if latents.ndim == 5:
            if denoise_indices is None:
                latents.squeeze_(1)
            else:
                latents = latents[:, denoise_indices]
                latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")

        images = self.decode_image(latents, *size)

        for i, img in enumerate(images):
            img.save(f"output_{i}.png")
