from pathlib import Path as _Path
from typing import List, Optional, Tuple, Union, Dict

import einops
import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from ...misc.image_utils import CenterCropResizeImage, get_closest_ratio
from ...pipeline import (
    DEFAULT_SETTINGS,
    Datatype,
    Images,
    ImageSettings,
    Path,
    Pipelinelike,
    Prompts,
    Pseudorandom,
    Schedulerlike,
    calculate_shift,
    requires,
    retrieve_timesteps,
)
from .model import NextDiT
from ...quant import quantize_model


class OneDiffusionPipeline(Pipelinelike):
    def __init__(
        self,
        transformer: NextDiT,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        scheduler: Schedulerlike,
    ) -> None:
        super().__init__()

        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.offload = False
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        self.max_length = 300

    @classmethod
    @torch.no_grad()
    def from_pretrained(
        cls,
        file_or_folder: Path,
        device: torch.device = "cpu",
        quantization_device: torch.device = "cuda",
        dtype: Union[Datatype, Dict[str, Datatype]] = torch.bfloat16,
    ) -> "OneDiffusionPipeline":
        if not isinstance(file_or_folder, _Path):
            file_or_folder = _Path(file_or_folder)
        datadict = {}
        if isinstance(dtype, dict):
            datadict = dtype
        else:
            datadict = {
                "transformer": dtype,
                "vae": dtype if isinstance(dtype, torch.dtype) else torch.bfloat16,
                "text_encoder": dtype,
            }

        if file_or_folder.is_dir():
            # diffusers-style
            with torch.device("meta"):
                transformer = NextDiT(
                    **NextDiT.load_config(file_or_folder / "transformer")
                )

            file = [
                x
                for x in (file_or_folder / "transformer").iterdir()
                if x.name.endswith(".pt") or x.name.endswith(".safetensors")
            ][0]

            if file.name.endswith(".safetensors"):
                datatype = datadict.get("transformer", torch.bfloat16)
                if isinstance(datatype, torch.dtype):
                    torch_dtype = datatype

                    from safetensors.torch import load_model

                    load_model(transformer, file, device=device)
                    transformer.to(dtype=torch_dtype)
                else:
                    torch_dtype = torch.bfloat16
                    dtype = datatype

                    transformer = cls.create_quantized_model_from_safetensors(
                        transformer,
                        file,
                        device=device,
                        quantization_device=quantization_device,
                        dtype=dtype,
                    )
            else:
                raise ValueError("Only .safetensors are supported for now")

            datatype = datadict.get("text_encoder", torch.bfloat16)
            text_encoder = T5EncoderModel.from_pretrained(
                file_or_folder / "text_encoder"
            )
            if isinstance(datatype, torch.dtype):
                text_encoder.to(dtype=datatype)
            else:
                torch_dtype = torch.bfloat16
                dtype = datatype

                text_encoder.to(dtype=torch_dtype)
                text_encoder = quantize_model(text_encoder, dtype, device=device)

            tokenizer = T5Tokenizer.from_pretrained(file_or_folder / "tokenizer")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                file_or_folder / "scheduler"
            )

            datatype = datadict.get("vae", torch.bfloat16)

            torch_dtype = torch.bfloat16
            if isinstance(datatype, torch.dtype):
                torch_dtype = datatype
            vae = AutoencoderKL.from_pretrained(file_or_folder / "vae").to(
                dtype=torch_dtype
            )

            ret = cls(transformer, vae, text_encoder, tokenizer, scheduler)
            ret.to(device=device)
            return ret
        else:
            raise ValueError("Single file loading not implemented for OneDiffusion")

    @requires("text_encoder")
    def prompt(
        self,
        prompts: Prompts,
        images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int]:
        dtype = torch.float32

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
        prompt_embeds = text_encoder_output[0].to(dtype)

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
            uncond_prompt_embeds = uncond_text_encoder_output[0].to(dtype)
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
        from ...misc.ray_utils import calculate_rays, create_c2w_matrix

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
        latents: torch.Tensor = None,
    ) -> Images:
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

        self.cfg.setup(steps, cfg, not do_cfg)
        self.postprocessors.setup(self)

        processed_image = self.encode_image(image, image_settings, *size)

        latents = self.prepare_latents(
            batch_size * images_per_prompt,
            self.transformer.config.in_channels,
            *size,
            generator=generator,
            image=processed_image,
            latents=latents,
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
        if self.can_mu:
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
        else:
            timesteps, steps = retrieve_timesteps(
                self.scheduler, steps, self.device, sigmas=sigmas
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

            noise_pred = self.cfg(latent_model_input, noise_pred, t, i)

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
        images = self.postprocessors(images, *size)
        return images
