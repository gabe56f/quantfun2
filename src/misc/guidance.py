import math
from typing import Tuple

import torch


class Guidance:
    def setup(self, steps: int, scale: float, disable: bool = False) -> "Guidance": ...

    def __call__(
        self,
        x0: torch.Tensor,
        conds: torch.Tensor,
        timestep: torch.LongTensor,
        step: int,
    ) -> torch.Tensor: ...


class CFG(Guidance):
    disable: bool = False
    scale: float = 1.0

    def setup(self, steps: int, scale: float, disable: bool = False) -> Guidance:
        self.disable = disable
        self.scale = scale

        return self

    def _cfg(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        return uncond + self.scale * (cond - uncond)

    def __call__(
        self,
        x0: torch.Tensor,
        conds: torch.Tensor,
        timestep: torch.LongTensor,
        step: int,
    ) -> torch.Tensor:
        if self.disable:
            return conds
        noise_pred_uncond, noise_pred_cond = conds.chunk(2)
        return self._cfg(noise_pred_cond, noise_pred_uncond)


class APG(Guidance):
    disable: bool = False
    steps: int = 0
    scale: float = 1.0

    momentum: float = 0.5
    adaptive_momentum: float = 0.18
    norm_threshold: float = 15.0
    eta: float = 1.0

    _signal_scale: float = momentum
    _running_average: float = 0

    def setup(self, steps: int, scale: float, disable: bool = False) -> Guidance:
        self.disable = disable
        self.steps = steps
        self.scale = scale

        # reset
        self._signal_scale = self.momentum
        self._running_average = 0

        return self

    def _update_buffer(self, update_value: torch.Tensor):
        new_avg = self.momentum * self._running_average
        self._running_average = update_value + new_avg

    def _project(
        self, v0: torch.Tensor, v1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = v0.dtype
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
        v0_ortho = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_ortho.to(dtype)

    def __call__(
        self,
        x0: torch.Tensor,
        conds: torch.Tensor,
        timestep: torch.LongTensor,
        step: int,
    ) -> torch.Tensor:
        if self.disable:
            return conds

        t = timestep.item()
        # print(t)
        self._signal_scale = self.momentum
        if self.adaptive_momentum is not None:
            if self.adaptive_momentum > 0:
                if self.momentum < 0:
                    self._signal_scale = (
                        -self.momentum * (self.adaptive_momentum**4) * (1000 - t)
                    )
                    if self._signal_scale > 0:
                        self._signal_scale = 0
                else:
                    self._signal_scale = self.momentum + (
                        self.adaptive_momentum**4
                    ) * (1000 - t)
                    if self._signal_scale < 0:
                        self._signal_scale = 0

        noise_pred_uncond, noise_pred_cond = conds.chunk(2)
        diff = noise_pred_cond - noise_pred_uncond

        self._update_buffer(diff)
        diff = self._running_average

        if self.norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, self.norm_threshold / diff_norm)
            diff = diff * scale_factor
        diff_parallel, diff_ortho = self._project(diff, noise_pred_cond)
        normalized_update = diff_ortho + self.eta * diff_parallel
        pred_guided = noise_pred_cond + (self.scale - 1) * normalized_update

        return pred_guided


class MimicCFG(Guidance):
    disable: bool = False
    scale: float = 1.0
    steps: int = 0

    mimic_scale: float = 1.0
    min_value: float = 0.0

    def setup(self, steps: int, scale: float, disable: bool = False) -> Guidance:
        self.disable = disable
        self.scale = scale
        self.steps = steps

        return self

    def _dyn_scale(self, scale: float, step: int) -> float:
        scale -= self.min_value
        max = self.steps - 1
        frac = step / max
        scale *= 1.0 - math.cos(frac)
        scale += self.min_value
        return scale

    def __call__(
        self,
        x0: torch.Tensor,
        conds: torch.Tensor,
        timestep: torch.LongTensor,
        step: int,
    ) -> torch.Tensor:
        if self.disable:
            return conds

        i = self.steps - step

        mimic_scale = self._dyn_scale(self.mimic_scale, i)
        cfg_scale = self._dyn_scale(self.scale, i)

        noise_pred_uncond, noise_pred_cond = conds.chunk(2)

        conds_per_batch = noise_pred_cond.shape[0] / noise_pred_uncond.shape[0]
        cond_stacked = noise_pred_cond.reshape(
            (-1, int(conds_per_batch)) + noise_pred_uncond.shape[1:]
        )

        diff = cond_stacked - noise_pred_uncond.unsqueeze(1)

        relative = diff.sum(1)

        mimic_target = noise_pred_uncond + relative * mimic_scale
        cfg_target = noise_pred_uncond + relative * cfg_scale

        def means_center(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            flattened = tensor.flatten(2)
            means = flattened.mean(dim=2).unsqueeze(2)
            centered = flattened - means
            return means, centered

        dt = noise_pred_cond.dtype
        cfg_means, cfg_centered = means_center(cfg_target)
        cfg_scaleref = (
            torch.quantile(cfg_centered.abs().to(torch.float32), 1.0, dim=2)
            .unsqueeze(2)
            .to(dt)
        )

        _, mimic_centered = means_center(mimic_target)
        mimic_scaleref = mimic_centered.abs().max(dim=2).values.unsqueeze(2)

        max_scaleref = torch.maximum(mimic_scaleref, cfg_scaleref)

        cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
        cfg_renormalized = (cfg_clamped / max_scaleref) * mimic_scaleref
        result = cfg_renormalized + cfg_means
        actual_res = result.unflatten(2, mimic_target.shape[2:])
        actual_res = actual_res * 0.7 + cfg_target * 0.3
        return actual_res


class CFGZero(Guidance):
    disable: bool = False
    scale: float = 1.0

    def setup(self, steps: int, scale: float, disable: bool = False) -> Guidance:
        self.disable = disable
        self.scale = scale

        return self

    def _optimize_scale(
        self, noise_pred_cond: torch.Tensor, noise_pred_uncond: torch.Tensor
    ) -> torch.Tensor:
        bs = noise_pred_uncond.shape[0]
        flat_cond = noise_pred_cond.view(bs, -1).to(torch.float32)
        flat_uncond = noise_pred_uncond.view(bs, -1).to(torch.float32)

        dot_product = torch.sum(flat_cond * flat_uncond, dim=1, keepdim=True)
        squared_norm = torch.sum(flat_uncond.square_(), dim=1, keepdim=True) + 1e-8
        dot_product /= squared_norm

        dot_product = dot_product.reshape([bs] + [1] * (noise_pred_cond.ndim - 1))
        return dot_product.to(noise_pred_cond.dtype)

    def _cfg(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        return uncond + self.scale * (cond - uncond)

    def __call__(
        self,
        x0: torch.Tensor,
        conds: torch.Tensor,
        timestep: torch.LongTensor,
        step: int,
    ) -> torch.Tensor:
        if self.disable:
            return conds

        noise_pred_uncond, noise_pred_cond = conds.chunk(2)
        diff_uncond, diff_cond = (x0 - conds).chunk(2)

        cfg = self._cfg(noise_pred_cond, noise_pred_uncond)
        alpha = self._optimize_scale(diff_cond, diff_uncond)
        alpha = (1.0 - alpha) * (self.scale - 1.0)
        correction = noise_pred_uncond * alpha

        return cfg + correction


class Mahiro(Guidance):
    disable: bool = False
    scale: float = 1.0

    def setup(self, steps: int, scale: float, disable: bool = False) -> Guidance:
        self.disable = disable
        self.scale = scale

        return self

    def _cfg(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        return uncond + self.scale * (cond - uncond)

    def __call__(
        self,
        x0: torch.Tensor,
        conds: torch.Tensor,
        timestep: torch.LongTensor,
        step: int,
    ) -> torch.Tensor:
        if self.disable:
            return conds

        noise_pred_uncond, noise_pred_cond = conds.chunk(2)

        cond_leap = noise_pred_cond * self.scale
        cfg = self._cfg(noise_pred_cond, noise_pred_uncond)
        merge = cond_leap + cfg / 2.0

        norm_uncond = (
            torch.sqrt((noise_pred_uncond * self.scale).abs())
            * noise_pred_uncond.sign()
        )
        norm_merge = torch.sqrt(merge.abs()) * merge.sign()
        sim = torch.nn.functional.cosine_similarity(norm_uncond, norm_merge).mean()
        alpha = (sim + 1.0) / 2.0

        return torch.lerp(cond_leap, cfg, alpha)


CFGS = {
    "cfg": CFG,
    "cfgzero": CFGZero,
    "mimic": MimicCFG,
    "apg": APG,
    "mahiro": Mahiro,
}
