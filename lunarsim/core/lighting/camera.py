"""Camera sensor model: fixed exposure (never auto -- the plan calls out that
lunar scenes have far too much dynamic range for auto-exposure to behave
sensibly) + shot/read noise + saturation, applied to a synthetic radiance
image so it's usable identically whether the image came from a fast
raster pass or a path-traced one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraNoiseModel:
    full_well_electrons: float = 20_000.0  # pixel saturation capacity
    read_noise_electrons: float = 5.0  # stddev, Gaussian
    quantum_efficiency: float = 0.6
    bit_depth: int = 12

    def apply(self, irradiance_image: np.ndarray, exposure_s: float, rng: np.random.Generator) -> np.ndarray:
        """`irradiance_image` is radiance/irradiance in arbitrary linear units
        proportional to photon flux; returns a quantized digital-number image
        in [0, 2**bit_depth - 1], clipped at saturation.
        """
        signal_electrons = irradiance_image * exposure_s * self.quantum_efficiency
        signal_electrons = np.clip(signal_electrons, 0, None)

        shot_noise = rng.poisson(signal_electrons).astype(np.float64)
        read_noise = rng.normal(0.0, self.read_noise_electrons, size=irradiance_image.shape)

        electrons = np.clip(shot_noise + read_noise, 0, self.full_well_electrons)

        max_dn = 2**self.bit_depth - 1
        dn = electrons / self.full_well_electrons * max_dn
        return np.round(np.clip(dn, 0, max_dn)).astype(np.uint16 if self.bit_depth <= 16 else np.uint32)


def apply_noise(
    irradiance_image: np.ndarray,
    exposure_s: float,
    rng: np.random.Generator,
    mode: str = "shot_read",
    **model_kwargs,
) -> np.ndarray:
    """`mode` matches the quality-profile `camera.noise` key: none | shot_read |
    shot_read_lens | full. `shot_read_lens`/`full` still apply shot+read here;
    the lens/vignette part is a per-pixel radiance falloff to be applied
    upstream by the renderer/adapter, not this sensor-noise stage.
    """
    if mode == "none":
        max_dn = 2 ** model_kwargs.get("bit_depth", 12) - 1
        model = CameraNoiseModel(**model_kwargs)
        signal_electrons = np.clip(irradiance_image * exposure_s * model.quantum_efficiency, 0, model.full_well_electrons)
        dn = signal_electrons / model.full_well_electrons * max_dn
        return np.round(np.clip(dn, 0, max_dn)).astype(np.uint16)

    model = CameraNoiseModel(**model_kwargs)
    return model.apply(irradiance_image, exposure_s, rng)
