from typing import Union
import math

import torch
import numpy as np


class Scheduler:
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...


class Sampler:
    num_inference_steps: int
    timesteps: torch.Tensor
    sigmas: torch.Tensor

    scheduler: Scheduler

    def __setattr__(self, __name: str, __value) -> None:
        if __name == "scheduler":
            if isinstance(__value, str):
                __value = SCHEDULERS[__value]()
        super().__setattr__(__name, __value)

    def _calculate_mu(self, image_seq_len: int) -> float:
        max_shift = getattr(self, "max_shift", None)
        base_shift = getattr(self, "base_shift", None)
        if max_shift is not None and base_shift is not None:
            # hardcode since, I don't think these change ANYWHERE...
            max_seq_len = 4096
            base_seq_len = 256

            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b

            return mu
        return None

    def scale_noise(self, z: torch.Tensor) -> torch.Tensor: ...

    def init_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        image_seq_len: int = None,
    ): ...

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
    ) -> tuple: ...


class LinearScheduler(Scheduler):
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.linspace(sigma_max, sigma_min, num_inference_steps, dtype=dtype)


class KarrasScheduler(Scheduler):
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rho = 7.0  # default in paper
        ramp = torch.linspace(0, 1, num_inference_steps, dtype=dtype)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


class ExponentialScheduler(Scheduler):
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.exp(
            torch.linspace(
                math.log(sigma_max),
                math.log(sigma_min),
                num_inference_steps,
                dtype=dtype,
            )
        )


class BetaScheduler(Scheduler):
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha = beta = 0.6

        import scipy.stats

        sigmas = torch.from_numpy(
            np.array(
                [
                    sigma_min + (ppf * (sigma_max - sigma_min))
                    for ppf in [
                        scipy.stats.beta.ppf(timestep, alpha, beta)
                        for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                    ]
                ]
            )
        ).to(dtype=dtype)
        return sigmas


SCHEDULERS = {
    "linear": LinearScheduler,
    "karras": KarrasScheduler,
    "exponential": ExponentialScheduler,
    "beta": BetaScheduler,
}


class FlowMatchEulerScheduler(Sampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        float32: bool = False,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift

        self.dtype = torch.float32 if float32 else torch.bfloat16
        self.schedule: Scheduler = LinearScheduler()

        self.timesteps = torch.linspace(
            1.0,
            num_train_timesteps,
            num_train_timesteps,
            dtype=self.dtype,
        )[::-1]

        self.sigmas = self.timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)

        self.timesteps = self.sigmas / num_train_timesteps

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.num_inference_steps = num_train_timesteps
        self._step_index = 0

    def scale_noise(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def init_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        image_seq_len: int = None,
    ):
        sigmas = torch.linspace(
            self.sigma_max, self.sigma_min, num_inference_steps, dtype=self.dtype
        )

        if self.use_dynamic_shifting and image_seq_len is not None:
            mu = self._calculate_mu(image_seq_len)
            sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))  # ** 1.0)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        sigmas = self.schedule.create_sigmas(
            num_inference_steps, sigmas[0].item(), sigmas[-1].item(), self.dtype
        )
        self.timesteps = (sigmas * self.num_train_timesteps).to(device)
        self.sigmas = torch.cat(
            [sigmas, torch.zeros(1, dtype=self.dtype, device=sigmas.device)]
        )
        self.num_inference_steps = num_inference_steps
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> tuple:
        sample = sample.to(self.dtype)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1
        return (prev_sample,)


SAMPLERS = {
    "euler": FlowMatchEulerScheduler,
}
