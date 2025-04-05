from typing import Union, Dict, List, Tuple
from pathlib import Path as _Path

import numpy as np
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import T5Tokenizer, T5EncoderModel
import torch
from tqdm.auto import tqdm
from PIL import Image

from .mmdit import FluxTransformer2D
from ...pipeline import (
    DEFAULT_SETTINGS,
    Datatype,
    Images,
    ImageSettings,
    Path,
    Pipelinelike,
    Prompts,
    Pseudorandom,
    requires,
    retrieve_timesteps,
    Sampler,
)
from ...quant import quantize_model


class FluxPipeline(Pipelinelike):
    def __init__(
        self,
        transformer: FluxTransformer2D,
        scheduler: Sampler,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
    ):
        super().__init__()

        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.offload = False
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

        self.vae_scale_factor = 8

        self.max_length = 256

    @classmethod
    @torch.no_grad()
    def from_pretrained(
        cls,
        file_or_folder: Path,
        device: torch.device = "cpu",
        quantization_device: torch.device = "cuda",
        dtype: Union[Datatype, Dict[str, Datatype]] = torch.bfloat16,
    ) -> Pipelinelike:
        raise NotImplementedError("from_pretrained not implemented yet")
        # if not isinstance(file_or_folder, _Path):
        #     file_or_folder = _Path(file_or_folder)
        # datadict = {}
        # if isinstance(dtype, dict):
        #     datadict = dtype
        # else:
        #     datadict = {
        #         "transformer": dtype,
        #         "vae": dtype if isinstance(dtype, torch.dtype) else torch.bfloat16,
        #         "text_encoder": dtype,
        #     }

        # if file_or_folder.is_dir():
        #     with torch.device("meta"):
        #         transformer = create_2b_model()
        #     file = [
        #         x
        #         for x in (file_or_folder / "transformer").iterdir()
        #         if x.name.endswith(".pt") or x.name.endswith(".safetensors")
        #     ][0]
        #     if file.name.endswith(".safetensors"):
        #         datatype = datadict.get("transformer", torch.bfloat16)
        #         if isinstance(datatype, torch.dtype):
        #             torch_dtype = datatype

        #             from safetensors.torch import load_file

        #             state_dict = load_file(file)
        #             state_dict = LuminaDiT.transform_state_dict(state_dict)
        #             transformer.load_state_dict(state_dict, assign=True, strict=True)
        #             transformer.to(dtype=torch_dtype, device=device)
        #         else:
        #             dtype = datatype
        #             # print(LuminaDiT.get_replace_map())
        #             transformer = cls.create_quantized_model_from_safetensors(
        #                 transformer,
        #                 file,
        #                 device=device,
        #                 quantization_device=quantization_device,
        #                 dtype=dtype,
        #                 replace_map=LuminaDiT.get_replace_map(),
        #             )
        #     else:
        #         raise ValueError("Only .safetensors is supported for now.")
        #     datatype = datadict.get("text_encoder", torch.bfloat16)
        #     text_encoder = Gemma2ForCausalLM.from_pretrained(
        #         file_or_folder / "text_encoder"
        #     )
        #     if isinstance(datatype, torch.dtype):
        #         text_encoder.to(dtype=datatype)
        #     else:
        #         dtype = datatype

        #         text_encoder = quantize_model(
        #             text_encoder,
        #             dtype,
        #             device=device,
        #             quantization_device=quantization_device,
        #         )
        #     tokenizer = GemmaTokenizer.from_pretrained(file_or_folder / "tokenizer")
        #     scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        #         file_or_folder / "scheduler"
        #     )

        #     datatype = datadict.get("vae", torch.bfloat16)
        #     torch_dtype = torch.bfloat16
        #     if isinstance(datatype, torch.dtype):
        #         torch_dtype = datatype
        #     vae = AutoencoderKL.from_pretrained(file_or_folder / "vae").to(
        #         dtype=torch_dtype
        #     )
        #     ret = cls(transformer, scheduler, vae, text_encoder, tokenizer)
        #     ret.to(device=device)
        #     return ret
        # else:
        #     raise ValueError("single file loading not done yet")

    @requires("text_encoder")
    def prompt(
        self, prompts: Prompts, images_per_prompt: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device

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
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder(
            text_input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        prompt_embeds = prompt_embeds.hidden_states[-2]

        bsz, seqlen, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bsz * images_per_prompt, seqlen, -1)
        attention_mask = attention_mask.repeat(images_per_prompt, 1)
        attention_mask = attention_mask.view(bsz * images_per_prompt, -1)

        if len(negative) > 0:
            uncond_text_inputs = self.tokenizer(
                negative,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_input_ids = uncond_text_inputs.input_ids.to(device=device)
            uncond_attention_mask = uncond_text_inputs.attention_mask.to(device=device)

            uncond_prompt_embeds = self.text_encoder(
                uncond_text_input_ids,
                attention_mask=uncond_attention_mask,
                output_hidden_states=True,
            )
            uncond_prompt_embeds = uncond_prompt_embeds.hidden_states[-2]

            bsz, seqlen, _ = uncond_prompt_embeds.shape
            uncond_prompt_embeds = uncond_prompt_embeds.repeat(1, images_per_prompt, 1)
            uncond_prompt_embeds = uncond_prompt_embeds.view(
                bsz * images_per_prompt, seqlen, -1
            )
            uncond_attention_mask = uncond_attention_mask.repeat(images_per_prompt, 1)
            uncond_attention_mask = uncond_attention_mask.view(
                bsz * images_per_prompt, -1
            )

            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
            attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        return (
            prompt_embeds.to(device, self.dtype),
            attention_mask.to(device, self.dtype),
            len(negative) > 0,
            bsz,
        )

    @requires("transformer")
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
        shape = (
            batch_size,
            latent_channels,
            2 * (height // (self.vae_scale_factor * 2)),
            2 * (width // (self.vae_scale_factor * 2)),
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
            latents = latents.to(dtype=self.dtype)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    @requires("vae")
    def decode_image(
        self, latents: torch.Tensor, height: int = 1024, width: int = 1024
    ) -> Images:
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor

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
        eta: float = 0,
        denoise_mask: List[int] = [1, 0],
        noise_scale: float = 1,
        latents: torch.Tensor = None,
    ) -> Images:
        if isinstance(seed, int):
            generator = torch.Generator(self.device)
            generator = generator.manual_seed(seed)
        else:
            generator = seed

        height, width = size
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator=generator, eta=eta)
        encoder_hidden_states, encoder_attention_mask, do_cfg, batch_size = self.prompt(
            prompts, images_per_prompt
        )

        self.cfg.setup(steps, cfg, not do_cfg)
        self.postprocessors.setup(self)

        latents = self.prepare_latents(
            batch_size * images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            generator,
            latents,
        )

        timesteps, steps = retrieve_timesteps(
            self.scheduler, steps, self.device, image_seq_len=latents.shape[1]
        )

        for i, t in tqdm(enumerate(timesteps), total=steps):
            latents_input = torch.cat([latents] * 2) if do_cfg else latents
            curr_t = t
            if not torch.is_tensor(curr_t):
                curr_t = torch.tensor([curr_t], dtype=torch.float64, device=self.device)
            else:
                curr_t = curr_t[None].to(self.device)

            curr_t = curr_t.expand(latents_input.shape[0])
            curr_t = 1 - curr_t / self.scheduler.config.num_train_timesteps

            noise_pred = self.transformer(
                x=latents_input,
                t=curr_t,
                cap_feats=encoder_hidden_states,
                cap_mask=encoder_attention_mask,
            )

            noise_pred = self.cfg(latents_input, noise_pred, t, i)
            noise_pred = -noise_pred
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        images = self.decode_image(latents, *size)
        images = self.postprocessors(images, *size)
        return images
