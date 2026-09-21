"""Fractional Brownian motion noise for large-scale hill/depression terrain."""
from __future__ import annotations

import numpy as np


def _generate_fractal_noise(
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int,
    hurst: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum of successively finer Perlin-style gradient noise octaves.

    `res` is the number of periods of the base grid along each axis;
    each octave doubles that frequency. Persistence is derived from
    `hurst` so the spectral falloff matches fBm with the given exponent.
    """
    def _perlin(shape: tuple[int, int], res: tuple[int, int]) -> np.ndarray:
        def f(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]]
        grid = grid.transpose(1, 2, 0) % 1

        angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(shape[0] // res[0], 0).repeat(shape[1] // res[1], 1)
        g10 = gradients[1:, 0:-1].repeat(shape[0] // res[0], 0).repeat(shape[1] // res[1], 1)
        g01 = gradients[0:-1, 1:].repeat(shape[0] // res[0], 0).repeat(shape[1] // res[1], 1)
        g11 = gradients[1:, 1:].repeat(shape[0] // res[0], 0).repeat(shape[1] // res[1], 1)

        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1.0
    total = 0.0
    for _ in range(octaves):
        r = (res[0] * frequency, res[1] * frequency)
        if shape[0] % r[0] or shape[1] % r[1]:
            break
        noise += amplitude * _perlin(shape, r)
        total += amplitude
        frequency *= 2
        amplitude *= 2 ** (-hurst)
    return noise / total if total > 0 else noise


def fbm_heightfield(
    n: int,
    amplitude_m: float,
    wavelength_m: float,
    res_m: float,
    hurst: float,
    rng: np.random.Generator,
    octaves: int = 6,
) -> np.ndarray:
    """Generate an (n, n) low-frequency hill/depression heightfield.

    `wavelength_m` sets the base periodicity in world units; `hurst`
    (0.6-0.9 typical) controls roughness persistence across octaves.
    """
    size_m = n * res_m
    base_periods = max(1, round(size_m / wavelength_m))

    # find an n that is divisible by base_periods * 2**(octaves-1)
    divisor = base_periods * (2 ** (octaves - 1))
    pad_n = int(np.ceil(n / divisor)) * divisor
    pad_n = max(pad_n, divisor)

    noise = _generate_fractal_noise(
        (pad_n, pad_n), (base_periods, base_periods), octaves, hurst, rng
    )
    noise = noise[:n, :n]
    peak = np.abs(noise).max()
    if peak > 0:
        noise = noise / peak
    return noise * amplitude_m
