"""Molecular planarity metrics.

Adapted from SINGROUP/SMARTERS (notebooks/utils/planarity.py), with handling of degenerate
inputs. Usage: ``python -m tools.analysis.planarity --self-test``.
"""

import numpy as np

__all__ = ['pca', 'planarity', 'planarity_of', 'plane_project', 'PLANARITY_FIELDS',
           'SMARTERS_THRESHOLDS']

# Plane-RMSD cutoffs (Angstrom) of the SMARTERS dataset variants.
SMARTERS_THRESHOLDS = (0.05, 0.1, 0.5)

PLANARITY_FIELDS = ('planarity_pca', 'planarity_rms', 'plane_rmsd')


def pca(coords):
    """Return ``(eigvals, eigvecs, centred)`` of the coordinate covariance, eigvals ascending."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    centred = coords - coords.mean(axis=0)
    cov = (centred.T @ centred) / n          # divide by N, as in SMARTERS
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs, centred


def planarity(eigvals, eigvecs, X):
    """Return ``(planarity_pca, planarity_rms, rmsd)``; percentages and Angstrom.

    NaN if all points coincide.
    """
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = eigvals.sum()
    in_plane = eigvals[0] + eigvals[1]
    if not np.isfinite(total) or total <= 0 or in_plane <= 0:
        return float('nan'), float('nan'), float('nan')

    planarity_pca = 100 * in_plane / total

    normal = eigvecs[:, 2]
    d = (X @ normal) / np.linalg.norm(normal)
    rmsd = float(np.sqrt(np.mean(d ** 2)))
    L = np.sqrt(in_plane)
    planarity_rms = 100 * (1 - rmsd / L)
    return float(planarity_pca), float(planarity_rms), rmsd


def planarity_of(coords):
    """Planarity of an ``(n, 3)`` coordinate array; NaN for fewer than 3 atoms."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'expected an (n, 3) coordinate array, got {coords.shape}')
    if coords.shape[0] < 3:
        return float('nan'), float('nan'), float('nan')
    return planarity(*pca(coords))


def plane_project(pos):
    """Project onto the best-fit plane; return ``(xy, out_of_plane_rms)``."""
    c = pos - pos.mean(0)
    if len(c) < 3:
        return c[:, :2], 0.0
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:2].T
    oop = c @ vt[2]
    return xy, float(np.sqrt((oop ** 2).mean()))


def _smarters_reference(coords):
    """Original SMARTERS implementation, used as the self-test reference."""
    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    centroid = coords.mean(axis=0)
    X = coords - centroid
    C = (X.T @ X) / N
    eigvals, eigvecs = np.linalg.eigh(C)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    planarity_pca = 100 * (eigvals[0] + eigvals[1]) / eigvals.sum()
    normal = eigvecs[:, 2]
    d = (X @ normal) / np.linalg.norm(normal)
    rmsd = np.sqrt(np.mean(d ** 2))
    L = np.sqrt(eigvals[0] + eigvals[1])
    planarity_rms = 100 * (1 - rmsd / L)
    return planarity_pca, planarity_rms, rmsd


def _benzene(plane='xy'):
    """Planar benzene in the xy or xz plane."""
    ang = np.arange(6) * np.pi / 3
    rc, rh = 1.39, 2.48
    xs = np.concatenate([rc * np.cos(ang), rh * np.cos(ang)])
    ys = np.concatenate([rc * np.sin(ang), rh * np.sin(ang)])
    zs = np.zeros(12)
    return np.stack([xs, ys, zs] if plane == 'xy' else [xs, zs, ys], axis=1)


def self_test():
    """Check against the SMARTERS reference and simple known cases."""
    ok = failed = 0

    def check(name, cond, detail=''):
        nonlocal ok, failed
        print(f'  {"PASS" if cond else "FAIL"}  {name}{"  " + detail if detail else ""}')
        if cond:
            ok += 1
        else:
            failed += 1

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(200):
        n = int(rng.integers(3, 60))
        squash = float(rng.choice([0.0, 0.01, 0.1, 0.5, 1.0]))
        pts = rng.normal(size=(n, 3)) * np.array([4.0, 3.0, squash])
        mine = planarity_of(pts)
        ref = _smarters_reference(pts)
        worst = max(worst, max(abs(a - b) for a, b in zip(mine, ref)))
    check('matches the SMARTERS reference on 200 random clouds', worst < 1e-9,
          f'worst abs difference {worst:.2e}')

    p_xy = planarity_of(_benzene('xy'))
    p_xz = planarity_of(_benzene('xz'))
    check('planar benzene: rmsd 0, both percentages 100',
          abs(p_xy[2]) < 1e-12 and abs(p_xy[0] - 100) < 1e-9 and abs(p_xy[1] - 100) < 1e-9,
          f'pca {p_xy[0]:.4f}%  rms {p_xy[1]:.4f}%  rmsd {p_xy[2]:.2e}')
    check('rotation invariant: benzene stood on edge scores identically',
          all(abs(a - b) < 1e-9 for a, b in zip(p_xy, p_xz)),
          f'xz: pca {p_xz[0]:.4f}%  rms {p_xz[1]:.4f}%  rmsd {p_xz[2]:.2e}')

    buckled = _benzene('xy').copy()
    buckled[:6, 2] = np.array([0.25, -0.25, 0.25, -0.25, 0.25, -0.25])
    pb = planarity_of(buckled)
    check('buckled ring: rmsd > 0 and both percentages drop',
          pb[2] > 0.05 and pb[0] < 100 and pb[1] < 100,
          f'pca {pb[0]:.2f}%  rms {pb[1]:.2f}%  rmsd {pb[2]:.3f} A')
    check('planarity_rms is the more sensitive of the two',
          (100 - pb[1]) > (100 - pb[0]),
          f'pca drops {100 - pb[0]:.2f} pts, rms drops {100 - pb[1]:.2f} pts')

    for name, pts in (('2 atoms', np.array([[0., 0, 0], [1, 0, 0]])),
                      ('all atoms coincident', np.zeros((5, 3)))):
        got = planarity_of(pts)
        check(f'{name} -> NaN, not 100%',
              all(np.isnan(v) for v in got), str(got))

    worst = 0.0
    for trial in range(50):
        n = int(rng.integers(3, 40))
        pts = rng.normal(size=(n, 3)) * np.array([3.0, 2.0, 0.3])
        _, oop = plane_project(pts)
        worst = max(worst, abs(oop - planarity_of(pts)[2]))
    check('plane_project out-of-plane RMS == planarity rmsd', worst < 1e-12,
          f'worst abs difference {worst:.2e}')

    check('SMARTERS thresholds are 0.05 / 0.1 / 0.5 A',
          SMARTERS_THRESHOLDS == (0.05, 0.1, 0.5), str(SMARTERS_THRESHOLDS))

    print(f'\nself-test: {ok} passed, {failed} failed')
    return ok, failed


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true', help='no data required')
    a = ap.parse_args()
    if a.self_test:
        _, failed = self_test()
        raise SystemExit(1 if failed else 0)
    ap.print_help()
