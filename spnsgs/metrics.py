from __future__ import annotations
import numpy as np
def rel2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm((pred-ref).ravel()) / max(np.linalg.norm(ref.ravel()), 1e-30))


def pearson(pred: np.ndarray, ref: np.ndarray) -> float | None:
    a, b = pred.ravel(), ref.ravel()
    if np.std(a) <= 1e-30 or np.std(b) <= 1e-30:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a*a)))


def event_metrics(pi: np.ndarray, ref: np.ndarray) -> dict:
    positive, truth = pi < 0, ref < 0
    tp = int(np.count_nonzero(positive & truth))
    fp = int(np.count_nonzero(positive & ~truth))
    fn = int(np.count_nonzero(~positive & truth))
    return {
        "P_Pi_negative": float(np.mean(positive)),
        "reference_P_Pi_negative": float(np.mean(truth)),
        "negative_precision": tp / (tp + fp) if tp + fp else None,
        "negative_recall": tp / (tp + fn) if tp + fn else None,
        "negative_F1": 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else None,
        "negative_IoU": tp / (tp + fp + fn) if tp + fp + fn else None,
    }


def tail(pi: np.ndarray, prefix: str) -> dict:
    negative = pi[pi < 0]
    if negative.size == 0:
        return {f"{prefix}_negative_p{p}": None for p in (1, 5, 10)} | {
            f"{prefix}_negative_conditional_mean": None}
    values = {f"{prefix}_negative_p{p}": float(np.percentile(negative, p)) for p in (1, 5, 10)}
    values[f"{prefix}_negative_conditional_mean"] = float(np.mean(negative))
    return values


def transfer_spectrum(omega: np.ndarray, div: np.ndarray) -> np.ndarray:
    n = omega.size
    co = np.real(np.fft.fft2(omega)/n * np.conj(np.fft.fft2(-div)/n))
    ky = np.fft.fftfreq(omega.shape[0], d=1/omega.shape[0])
    kx = np.fft.fftfreq(omega.shape[1], d=1/omega.shape[1])
    shell = np.rint(np.hypot(ky[:, None], kx[None, :])).astype(int)
    return np.bincount(shell.ravel(), weights=co.ravel())


def spectral_l1(pred: np.ndarray, ref: np.ndarray) -> float:
    stop = min(len(pred), len(ref), 34)
    return float(np.sum(np.abs(pred[1:stop]-ref[1:stop])) /
                 max(np.sum(np.abs(ref[1:stop])), 1e-30))

def _radial_spectrum(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = u.shape
    energy = 0.5 * (
        np.abs(np.fft.fft2(u) / (nx * ny)) ** 2
        + np.abs(np.fft.fft2(v) / (nx * ny)) ** 2
    )
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    shell = np.rint(np.sqrt(kkx**2 + kky**2)).astype(np.int64)
    spectrum = np.bincount(shell.ravel(), weights=energy.ravel())
    return np.arange(spectrum.size), spectrum


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "maximum": float("nan"), "final": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "final": float(array[-1]),
    }

