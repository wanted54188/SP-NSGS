from __future__ import annotations

import numpy as np


def gaussian_filter_gain(wave_number: float, delta: float) -> float:
    """Top-hat-equivalent Gaussian gain exp(-delta^2*k^2/24)."""
    return float(np.exp(-(delta * wave_number) ** 2 / 24.0))


def gaussian_filter_2d(
    field: np.ndarray,
    dx: float,
    dy: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter a periodic scalar field and return the filtered x/y derivatives."""
    ny, nx = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    transfer = np.exp(-(delta**2) * (kkx**2 + kky**2) / 24.0)
    spectrum = np.fft.fft2(field)
    filtered_spectrum = transfer * spectrum
    filtered = np.fft.ifft2(filtered_spectrum).real
    derivative_x = np.fft.ifft2(1j * kkx * filtered_spectrum).real
    derivative_y = np.fft.ifft2(1j * kky * filtered_spectrum).real
    return filtered, derivative_x, derivative_y


def periodic_streamfunction_from_vorticity(
    omega: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Solve -laplacian(psi)=omega on a periodic grid with zero-mean psi."""
    ny, nx = omega.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    wave_number_squared = kkx**2 + kky**2
    omega_spectrum = np.fft.fft2(omega - np.mean(omega))
    psi_spectrum = np.zeros_like(omega_spectrum)
    nonzero = wave_number_squared > 0.0
    psi_spectrum[nonzero] = (
        omega_spectrum[nonzero] / wave_number_squared[nonzero]
    )
    return np.fft.ifft2(psi_spectrum).real


def periodic_second_derivatives(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xx, xy and yy derivatives on a periodic grid."""
    ny, nx = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    spectrum = np.fft.fft2(field)
    xx = np.fft.ifft2(-(kkx**2) * spectrum).real
    xy = np.fft.ifft2(-(kkx * kky) * spectrum).real
    yy = np.fft.ifft2(-(kky**2) * spectrum).real
    return xx, xy, yy


def periodic_divergence(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return div(flux) on a periodic Cartesian grid."""
    _, flux_x_x, _ = gaussian_filter_2d(flux_x, dx, dy, 0.0)
    _, _, flux_y_y = gaussian_filter_2d(flux_y, dx, dy, 0.0)
    return flux_x_x + flux_y_y


def filter_vorticity_snapshot(
    u: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    dx: float,
    dy: float,
    delta: float,
) -> dict[str, np.ndarray]:
    """Build filtered state and exact SGS vorticity flux from one raw snapshot."""
    u_bar, u_x, u_y = gaussian_filter_2d(u, dx, dy, delta)
    v_bar, v_x, v_y = gaussian_filter_2d(v, dx, dy, delta)
    omega_bar, omega_x, omega_y = gaussian_filter_2d(
        omega, dx, dy, delta
    )
    uomega_bar, _, _ = gaussian_filter_2d(u * omega, dx, dy, delta)
    vomega_bar, _, _ = gaussian_filter_2d(v * omega, dx, dy, delta)
    q_sgs_x = uomega_bar - u_bar * omega_bar
    q_sgs_y = vomega_bar - v_bar * omega_bar
    pi_sgs = -(q_sgs_x * omega_x + q_sgs_y * omega_y)
    div_q_sgs = periodic_divergence(q_sgs_x, q_sgs_y, dx, dy)
    omega_xx, omega_xy, omega_yy = periodic_second_derivatives(
        omega_bar, dx, dy
    )
    return {
        "u": u_bar,
        "v": v_bar,
        "omega": omega_bar,
        "omega_x": omega_x,
        "omega_y": omega_y,
        "omega_xx": omega_xx,
        "omega_xy": omega_xy,
        "omega_yy": omega_yy,
        "u_x": u_x,
        "u_y": u_y,
        "v_x": v_x,
        "v_y": v_y,
        "psi": periodic_streamfunction_from_vorticity(
            omega_bar, dx, dy
        ),
        "q_sgs_x": q_sgs_x,
        "q_sgs_y": q_sgs_y,
        "pi_sgs": pi_sgs,
        "div_q_sgs": div_q_sgs,
    }
