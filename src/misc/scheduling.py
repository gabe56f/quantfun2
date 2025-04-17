from typing import Callable, Union, Literal, List, Optional, Tuple
import math

import torch
from tqdm.auto import tqdm
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
    num_train_timesteps: int

    timesteps: torch.Tensor
    sigmas: torch.Tensor
    scheduler: Scheduler

    def __setattr__(self, __name: str, __value) -> None:
        if __name == "scheduler":
            if isinstance(__value, str):
                __value = SCHEDULERS[__value]()
        super().__setattr__(__name, __value)

    @property
    def steps(self) -> int:
        return self.timesteps.shape[0]

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
    ) -> torch.Tensor:
        """Not necessarily implemented."""
        ...

    def sample(
        self,
        x0: torch.Tensor,
        func: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor],
        num_inference_steps: int,
        device: torch.device,
        image_seq_len: int = None,
    ) -> torch.Tensor: ...


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


class KLOptimalScheduler(Scheduler):
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        adj_idxs = torch.arange(num_inference_steps, dtype=dtype).div_(
            num_inference_steps - 1
        )
        return (
            adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)
        ).tan_()


class PolyexponentialScheduler(Scheduler):
    def create_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float,
        sigma_max: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rho = (1.0,)
        ramp = torch.linspace(1, 0, num_inference_steps, dtype=dtype) ** rho
        return torch.exp(
            ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min)
        )


SCHEDULERS = {
    "linear": LinearScheduler,
    "karras": KarrasScheduler,
    "exponential": ExponentialScheduler,
    "polyexponential": PolyexponentialScheduler,
    "kl-optimal": KLOptimalScheduler,
}


try:
    import scipy.stats

    class BetaScheduler(Scheduler):
        def create_sigmas(
            self,
            num_inference_steps: int,
            sigma_min: float,
            sigma_max: float,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            alpha = beta = 0.6

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

    SCHEDULERS["beta"] = BetaScheduler

except ImportError:
    pass


class FlowMatchEulerSampler(Sampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift

        self.dtype = dtype
        self.scheduler: Scheduler = LinearScheduler()

        timesteps = torch.linspace(
            num_train_timesteps,
            1.0,
            num_train_timesteps,
            dtype=self.dtype,
        )

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self.sigmas = sigmas.to("cpu")
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

        sigmas = self.scheduler.create_sigmas(
            num_inference_steps, sigmas[-1].item(), sigmas[0].item(), self.dtype
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
    ) -> torch.Tensor:
        sample = sample.to(self.dtype)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1
        return prev_sample

    def sample(
        self,
        x0: torch.Tensor,
        func: callable,
        num_inference_steps: int,
        device: torch.device,
        image_seq_len: int = None,
    ) -> torch.Tensor:
        self.init_timesteps(num_inference_steps, device, image_seq_len)

        for i, t in tqdm(enumerate(self.timesteps), total=self.steps):
            output = func(x0, i, t)
            x0 = self.step(output, t, x0)
        return x0


class StochasticEulerSampler(Sampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 6.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        score_interpolant: Literal["linear", "vp"] = "vp",
        g_type: Literal["paper", "zero"] = "paper",
        g_scale: float = 3.0,
        beta_0: float = 0.1,
        beta_1: float = 20.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_shift = base_shift
        self.max_shift = max_shift

        self.dtype = dtype
        self.scheduler: Scheduler = LinearScheduler()

        self.timesteps = torch.linspace(
            num_train_timesteps,
            1.0,
            num_train_timesteps,
            dtype=self.dtype,
        )

        self.sigmas = self.timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)

        self.timesteps = self.sigmas / num_train_timesteps

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.stability_eps = 1e-9
        self.score_interpolant = score_interpolant
        self.g_type = g_type
        self.g_scale = g_scale
        self.beta_0 = beta_0
        self.beta_1 = beta_1

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

        sigmas = self.scheduler.create_sigmas(
            num_inference_steps, sigmas[-1].item(), sigmas[0].item(), self.dtype
        )
        self.timesteps = (sigmas * self.num_train_timesteps).to(device)
        self.sigmas = torch.cat(
            [sigmas, torch.zeros(1, dtype=self.dtype, device=sigmas.device)]
        )
        self.num_inference_steps = num_inference_steps
        self._step_index = 0

    def _get_linear_interpolant(self, timestep: float | torch.Tensor) -> tuple:
        alpha_t = (1.0 - timestep).clamp(min=self.stability_eps)
        alpha_dot_p = -torch.ones_like(timestep)
        return alpha_t, alpha_dot_p

    def _get_vp_interpolant(self, timestep: float | torch.Tensor) -> tuple:
        input_ndim = timestep.ndim
        if input_ndim == 0:
            timestep = timestep.unsqueeze(0)

        log_alpha_bar_t = (
            -0.25 * timestep**2 * (self.beta_1 - self.beta_0)
            - 0.5 * timestep * self.beta_0
        )
        alpha_bar_t = torch.exp(log_alpha_bar_t)
        beta_t = self.beta_0 + timestep * (self.beta_1 - self.beta_0)

        alpha_t = torch.sqrt(alpha_bar_t.clamp(min=self.stability_eps**2))
        alpha_dot_t: torch.Tensor = -0.5 * beta_t * alpha_t

        # squeeze if timestep was scalar
        if input_ndim == 0:
            return alpha_t.squeeze(0), alpha_dot_t.squeeze(0)
        return alpha_t, alpha_dot_t

    def _get_diffusion_coefficient(self, timestep: float | torch.Tensor) -> float:
        if self.g_type == "paper":
            return self.g_scale * (timestep**2)
        else:
            return torch.zeros_like(timestep)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = sample.dtype
        sample = sample.to(self.dtype)
        model_output = model_output.to(self.dtype)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        # shouldn't happen but better safe than sorry
        dt = (sigma_next - sigma).clamp(min=0)
        if sigma < self.stability_eps:  # for stability
            self._step_index += 1
            return sample
        if self.score_interpolant == "linear":
            alpha_t, alpha_dot_p = self._get_linear_interpolant(timestep)
        else:
            alpha_t, alpha_dot_p = self._get_vp_interpolant(timestep)

        # ∇log p_t(xt) = (alpha_t * u_t(xt) - alpha_dot_t * xt) / sigma_t^2
        nabla_log_pt = (alpha_t * model_output - alpha_dot_p * sample) / (
            sigma**2 + self.stability_eps
        )
        gt = self._get_diffusion_coefficient(timestep)

        # f_t(xt) = u_t(xt) - (g_t^2 / 2) * ∇log p_t(xt)
        ft = model_output - (gt**2 / 2.0) * nabla_log_pt

        # x_{t-Δt} ≈ x_t - f_t(x_t)Δt + g_t * sqrt(Δt) * z , where z ~ N(0,I)
        # note the sign: step is x_current -> x_next = x_current - f_t*dt + noise
        prev_sample = sample - ft * dt

        if gt > self.stability_eps:  # if diffusion coefficient is >0, add noise
            noise = torch.randn_like(sample)
            prev_sample += gt * torch.sqrt(dt) * noise

        # convert back to original dtype
        prev_sample = prev_sample.to(original_dtype)
        self._step_index += 1
        return prev_sample


class DiffusersFlowMatchEulerSampler(Sampler):
    def __init__(self) -> None:
        super().__init__()

        from diffusers import UniPCMultistepScheduler

        self.sampler = UniPCMultistepScheduler(
            flow_shift=6.0,
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            solver_order=2,
        )
        self.num_train_timesteps = 1000

    def init_timesteps(
        self, num_inference_steps: int, device: torch.device, image_seq_len: int = None
    ):
        self.sampler.set_timesteps(num_inference_steps)
        self.timesteps = self.sampler.timesteps
        self.sigmas = self.sampler.sigmas
        self.num_inference_steps = num_inference_steps

    def scale_noise(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        ret = self.sampler.step(model_output, timestep, sample)
        return ret.prev_sample

    def sample(
        self,
        x0: torch.Tensor,
        func: callable,
        num_inference_steps: int,
        device: torch.device,
        image_seq_len: int = None,
    ) -> torch.Tensor:
        self.init_timesteps(num_inference_steps, device, image_seq_len)

        for i, t in tqdm(enumerate(self.timesteps), total=num_inference_steps):
            output = func(x0, i, t)
            x0 = self.step(output, t, x0)
        return x0


class FlowMatchNOrderSampler(FlowMatchEulerSampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        dtype: torch.dtype = torch.float32,
        order: int = 2,
    ) -> None:
        super().__init__(
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            base_shift,
            max_shift,
            dtype,
        )

        self.order: int = order
        self.stage: int = 0
        # f_0 is x0, f_x is sampler output for order x
        self.f: List[Optional[torch.Tensor]] = [None] * (order + 1)
        # delta timestep (sigma in practice)
        self.dt: torch.Tensor = None

    def _step(self):
        self._step_index += 1
        self.stage = (self.stage + 1) % self.order

    def init_timesteps(
        self, num_inference_steps: int, device: torch.device, image_seq_len: int = None
    ):
        # reset
        self.stage = 0
        self.f = [None] * (self.order + 1)
        self.dt = None

        # approximate amount of steps
        num_inference_steps = num_inference_steps // self.order + 1

        super().init_timesteps(num_inference_steps, device, image_seq_len)

        self.sigmas = torch.cat(
            [
                self.sigmas[0].unsqueeze(0),
                self.sigmas[1:-1].repeat_interleave(self.order),
                self.sigmas[-1].unsqueeze(0),
            ]
        )
        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps


class FlowMatchHeunSampler(FlowMatchNOrderSampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            base_shift,
            max_shift,
            dtype,
            order=2,
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        sample = sample.to(self.dtype)

        if self.stage == 0:
            sigma = self.sigmas[self._step_index]
            self.dt = self.sigmas[self._step_index + 1] - sigma
            self.f[0] = sample
        else:
            sigma = self.sigmas[self._step_index]

        denoised = sample - model_output * sigma
        derivative = (sample - denoised) / sigma
        self.f[self.stage + 1] = derivative

        if self.stage == 1:
            derivative = (self.f[1] + self.f[2]) / 2

        prev_sample = self.f[0] + derivative * self.dt
        prev_sample = prev_sample.to(model_output.dtype)
        self._step()

        return prev_sample


class FlowMatchRalstonSampler(FlowMatchNOrderSampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            base_shift,
            max_shift,
            dtype,
            order=2,
        )

        self.dt = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        sample = sample.to(self.dtype)

        if self.stage == 0:
            sigma = self.sigmas[self._step_index]
            self.dt = self.sigmas[self._step_index + 1] - sigma
            self.f[0] = sample
        else:
            sigma = self.sigmas[self._step_index]

        denoised = sample - model_output * sigma
        derivative = (sample - denoised) / sigma
        self.f[self.stage + 1] = derivative

        if self.stage == 0:
            prev_sample = sample + (2.0 / 3.0) * self.dt * self.f[1]
        else:
            derivative_diff = 0.25 * self.f[1] + 0.75 * self.f[2]
            prev_sample = self.f[0] + self.dt * derivative_diff

        prev_sample = prev_sample.to(model_output.dtype)
        self._step()

        return prev_sample


# actually ssprk3
class FlowMatchRK3Sampler(FlowMatchNOrderSampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            base_shift,
            max_shift,
            dtype,
            order=3,
        )

        self.dt = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        sample = sample.to(self.dtype)

        self.f[self.stage + 1] = model_output
        if self.stage == 0:
            self.dt = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
            self.f[0] = sample

            prev_sample = self.f[0] + self.dt * self.f[1]
        elif self.stage == 1:
            fourth_dt = 0.25 * self.dt
            prev_sample = self.f[0] + fourth_dt * (self.f[1] + self.f[2])
        else:
            sixth_dt = (1 / 6) * self.dt
            prev_sample = self.f[0] + sixth_dt * (self.f[1] + self.f[2] + 4 * self.f[3])
        prev_sample = prev_sample.to(model_output.dtype)
        self._step()

        return prev_sample


# TODO: fix?
class FlowMatchUniPCSampler(FlowMatchEulerSampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        solver_type: Literal["bh1", "bh2", "vary"] = "bh2",
        solver_order: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            base_shift,
            max_shift,
            dtype,
        )

        self.order = solver_order
        self.solver_type: str = solver_type
        self.last_sample: torch.Tensor = None
        self.lower_order_nums: int = 0
        self.lower_order_final: bool = True

    def init_timesteps(
        self, num_inference_steps: int, device: torch.device, image_seq_len: int = None
    ):
        self.last_sample = None
        self.lower_order_nums = 0
        self.f = [None] * self.order

        self.dt = None
        return super().init_timesteps(num_inference_steps, device, image_seq_len)

    def _normalize(self, device: torch.device, predictor: bool = False) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        if predictor:
            sigma_t = self.sigmas[self._step_index + 1]
            sigma_s0 = self.sigmas[self._step_index]
        else:
            sigma_t = self.sigmas[self._step_index]
            sigma_s0 = self.sigmas[self._step_index - 1]

        lambda_t = torch.log((1 - sigma_t) / sigma_t)
        lambda_s0 = torch.log((1 - sigma_s0) / sigma_s0)

        h = lambda_t - lambda_s0
        rks = []
        D1s = []
        for i in range(1, self.this_order):
            if predictor:
                si = self._step_index + i
            else:
                si = self._step_index - (i + 1)
            mi = self.f[-(i + 1)]

            sigma_si = self.sigmas[si]
            lambda_si = torch.log((1 - sigma_si) / sigma_si)

            rk = (lambda_si - lambda_s0) / h

            rks.append(rk)
            D1s.append((mi - self.f[-1]) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1
        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)

        for i in range(1, self.this_order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None
        return R, b, sigma_t, sigma_s0, h_phi_1, D1s, B_h

    def _corrector(
        self,
        model_output: torch.Tensor,
        last_sample: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        R, b, sigma_t, sigma_s0, h_phi_1, D1s, B_h = self._normalize(sample.device)

        if self.this_order == 1:
            rhos_c = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(sample.device).to(sample.dtype)

        x_t_ = sigma_t / sigma_s0 * last_sample - (1 - sigma_t) * h_phi_1 * self.f[-1]
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_output - self.f[-1]

        sample = x_t_ - (1 - sigma_t) * B_h * (corr_res + rhos_c[-1] * D1_t)
        return sample

    def _predictor(self, sample: torch.Tensor) -> torch.Tensor:
        R, b, sigma_t, sigma_s0, h_phi_1, D1s, B_h = self._normalize(
            sample.device, True
        )
        x_t_ = sigma_t / sigma_s0 * sample - (1 - sigma_t) * h_phi_1 * self.f[-1]
        if D1s is not None:
            if self.this_order == 2:
                rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
            else:
                rhos_p = (
                    torch.linalg.solve(R[:-1, :-1], b[:-1])
                    .to(sample.device)
                    .to(sample.dtype)
                )
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        sample = x_t_ - (1 - sigma_t) * B_h * pred_res
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = model_output.dtype
        sample = sample.to(self.dtype)
        model_output = model_output.to(self.dtype)

        sigma_t = self.sigmas[self._step_index]
        model_output = sample - sigma_t * model_output

        if self.last_sample is not None:
            sample = self._corrector(
                model_output,
                self.last_sample,
                sample,
            )

        for i in range(self.order - 1):
            self.f[i] = self.f[i + 1]
        self.f[-1] = model_output

        self.this_order = min(
            self.order,
            len(self.timesteps) - self._step_index,
            self.lower_order_nums + 1,
        )

        self.last_sample = sample
        prev_sample = self._predictor(sample)

        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1
        self._step_index += 1

        return prev_sample.to(output_dtype)


SAMPLERS = {
    "euler": FlowMatchEulerSampler,
    "euler-sde": StochasticEulerSampler,
    "diffusers-test": DiffusersFlowMatchEulerSampler,
    # n-order samplers
    "heun": FlowMatchHeunSampler,
    "ralston": FlowMatchRalstonSampler,
    "rk3": FlowMatchRK3Sampler,
    "unipc": FlowMatchUniPCSampler,
}
