"""How flat a molecule is, following SINGROUP/SMARTERS.

One implementation, because there were already two. ``tools/visualize_geometry.py:64-72``
and ``tools/analyze_stm.py`` both fitted a best-fit plane by SVD and reported the
out-of-plane RMS; a third copy is what docs/06-code-practices.md 1 exists to prevent, so
both now call in here instead.

Provenance (ladder in docs/README.md):
  * ``pca`` / ``planarity``  -- **ADAPT** of ``SINGROUP/SMARTERS``,
        ``notebooks/utils/planarity.py``.  The arithmetic is upstream's, transcribed
        unchanged so that numbers computed here are directly comparable to SMARTERS-derived
        datasets.  Two details of it are load-bearing and easy to "clean up" wrongly:

          - the covariance is ``(X.T @ X) / N`` -- divided by **N, not N-1**.  With N-1 the
            eigenvalues, and therefore ``L``, come out larger by ``N/(N-1)``, which shifts
            ``planarity_rms`` by a few tenths of a percent for a 20-atom molecule.  Small,
            but it would mean our percentages are not on the same scale as SMARTERS'.
          - ``L = sqrt(eigvals[0] + eigvals[1])`` is an **in-plane RMS extent**, not a
            maximum extent, so ``planarity_rms`` is not bounded below by 0.

  * ``plane_project``        -- **REUSE**, moved verbatim from
        ``tools/visualize_geometry.py:64-72``, signature unchanged, so that module becomes
        an import rather than a copy.
  * ``planarity_of``, the degeneracy guards, ``self_test`` -- **NEW**.

What the three numbers mean
---------------------------
For centred coordinates ``X`` with principal eigenvalues ``l0 >= l1 >= l2``:

``rmsd``          RMS distance of the atoms from their own best-fit plane, in **Angstrom**.
                  This is the number SMARTERS names its dataset variants after (its README
                  lists 0.05 / 0.1 / 0.5).  Rotation-invariant, so it is a property of the
                  molecule and not of how it happens to be oriented in the file.
``planarity_pca``  ``100 * (l0 + l1) / (l0 + l1 + l2)`` -- the share of positional variance
                  that lies in the plane.  Saturates fast: a molecule can sit at 99 % here
                  and still be visibly buckled, because ``l2`` is a *squared* quantity.
``planarity_rms``  ``100 * (1 - rmsd / L)`` -- the out-of-plane RMS as a fraction of the
                  in-plane RMS extent.  Scale-free, and far more discriminating than
                  ``planarity_pca`` in the regime this corpus actually occupies (MEASURED on
                  4000 STM val molecules: median planarity_pca 94.6 %, median planarity_rms
                  76.1 %).

Usage
-----
    python3 tools/planarity.py --self-test
"""

import numpy as np

__all__ = ['pca', 'planarity', 'planarity_of', 'plane_project', 'PLANARITY_FIELDS',
           'SMARTERS_THRESHOLDS']

# The three RMSD cut-offs SMARTERS' README names its dataset variants after, in Angstrom.
# Kept here as the default so every caller reports against the same ladder.
SMARTERS_THRESHOLDS = (0.05, 0.1, 0.5)

PLANARITY_FIELDS = ('planarity_pca', 'planarity_rms', 'plane_rmsd')


def pca(coords):
    """Eigen-decomposition of the coordinate covariance.  ADAPT of SMARTERS ``pca``.

    Returns ``(eigvals, eigvecs, X)`` with ``X`` the centred coordinates.  Eigenvalues come
    back ascending, as ``numpy.linalg.eigh`` always does; ``planarity`` re-sorts them.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    centred = coords - coords.mean(axis=0)
    cov = (centred.T @ centred) / n          # N, not N-1 -- see the module docstring
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs, centred


def planarity(eigvals, eigvecs, X):
    """``(planarity_pca, planarity_rms, rmsd)``.  ADAPT of SMARTERS ``planarity``.

    Upstream's arithmetic, plus guards on the two divisions it does not guard.  Upstream
    can divide by zero here (a single atom, or every atom at the same point, makes every
    eigenvalue 0); this returns NaN for those instead of a warning and an ``inf``.
    """
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = eigvals.sum()
    in_plane = eigvals[0] + eigvals[1]
    if not np.isfinite(total) or total <= 0 or in_plane <= 0:
        return float('nan'), float('nan'), float('nan')

    planarity_pca = 100 * in_plane / total

    normal = eigvecs[:, 2]                    # eigenvector of the smallest eigenvalue
    d = (X @ normal) / np.linalg.norm(normal)
    rmsd = float(np.sqrt(np.mean(d ** 2)))
    L = np.sqrt(in_plane)
    planarity_rms = 100 * (1 - rmsd / L)
    return float(planarity_pca), float(planarity_rms), rmsd


def planarity_of(coords):
    """``(planarity_pca, planarity_rms, rmsd)`` straight from coordinates.

    **Deviation from upstream, deliberate:** fewer than 3 atoms returns NaN rather than a
    number.  Upstream would return 100 % for a diatomic, which is *true* -- two points are
    trivially coplanar -- but it is not a measurement, and letting it into a percentile
    piles degenerate cases onto the "perfectly planar" end of the distribution where they
    look like evidence.  MEASURED: 4 of 4000 STM val records have fewer than 3 atoms.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'expected an (n, 3) coordinate array, got {coords.shape}')
    if coords.shape[0] < 3:
        return float('nan'), float('nan'), float('nan')
    return planarity(*pca(coords))


def plane_project(pos):
    """Project onto the molecule's own best-fit plane; also return out-of-plane RMS.

    REUSE, verbatim from tools/visualize_geometry.py:64-72 -- same signature, same return,
    so that module's ``flat {oop:.2f} A`` annotations do not move.  Equivalent to the
    ``rmsd`` of ``planarity_of``: for an (n, 3) matrix the third right-singular vector and
    the smallest-eigenvalue eigenvector of ``X.T @ X`` are the same direction.  Asserted in
    ``self_test``.
    """
    c = pos - pos.mean(0)
    if len(c) < 3:
        return c[:, :2], 0.0
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:2].T
    oop = c @ vt[2]
    return xy, float(np.sqrt((oop ** 2).mean()))


def _smarters_reference(coords):
    """Upstream's code, transcribed literally, used ONLY as the self-test oracle.

    Kept byte-faithful to ``SINGROUP/SMARTERS notebooks/utils/planarity.py`` -- including
    the unguarded divisions -- so the test compares our version against upstream rather
    than against itself.  Never called by the library path.
    """
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
    """Planar benzene, in the xy plane or stood up in the xz plane."""
    ang = np.arange(6) * np.pi / 3
    rc, rh = 1.39, 2.48
    xs = np.concatenate([rc * np.cos(ang), rh * np.cos(ang)])
    ys = np.concatenate([rc * np.sin(ang), rh * np.sin(ang)])
    zs = np.zeros(12)
    return np.stack([xs, ys, zs] if plane == 'xy' else [xs, zs, ys], axis=1)


def self_test():
    """Checks with answers known by hand, plus an oracle comparison against upstream."""
    ok = failed = 0

    def check(name, cond, detail=''):
        nonlocal ok, failed
        print(f'  {"PASS" if cond else "FAIL"}  {name}{"  " + detail if detail else ""}')
        if cond:
            ok += 1
        else:
            failed += 1

    # 1. The oracle. This is the check that matters: it compares our implementation against
    #    upstream's own arithmetic on random point clouds of varying flatness, so any future
    #    "tidy-up" that changes the formula (N vs N-1, max vs RMS extent) is caught.
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

    # 2. Benzene: exactly planar, and the answer must not depend on its orientation --
    #    which is what makes the metric a property of the molecule.
    p_xy = planarity_of(_benzene('xy'))
    p_xz = planarity_of(_benzene('xz'))
    check('planar benzene: rmsd 0, both percentages 100',
          abs(p_xy[2]) < 1e-12 and abs(p_xy[0] - 100) < 1e-9 and abs(p_xy[1] - 100) < 1e-9,
          f'pca {p_xy[0]:.4f}%  rms {p_xy[1]:.4f}%  rmsd {p_xy[2]:.2e}')
    check('rotation invariant: benzene stood on edge scores identically',
          all(abs(a - b) < 1e-9 for a, b in zip(p_xy, p_xz)),
          f'xz: pca {p_xz[0]:.4f}%  rms {p_xz[1]:.4f}%  rmsd {p_xz[2]:.2e}')

    # 3. A buckled ring must score strictly worse on BOTH percentages, and planarity_rms
    #    must move more than planarity_pca -- the reason both are reported.
    buckled = _benzene('xy').copy()
    buckled[:6, 2] = np.array([0.25, -0.25, 0.25, -0.25, 0.25, -0.25])
    pb = planarity_of(buckled)
    check('buckled ring: rmsd > 0 and both percentages drop',
          pb[2] > 0.05 and pb[0] < 100 and pb[1] < 100,
          f'pca {pb[0]:.2f}%  rms {pb[1]:.2f}%  rmsd {pb[2]:.3f} A')
    check('planarity_rms is the more sensitive of the two',
          (100 - pb[1]) > (100 - pb[0]),
          f'pca drops {100 - pb[0]:.2f} pts, rms drops {100 - pb[1]:.2f} pts')

    # 4. Degeneracy: upstream divides by zero here; we return NaN.
    for name, pts in (('2 atoms', np.array([[0., 0, 0], [1, 0, 0]])),
                      ('all atoms coincident', np.zeros((5, 3)))):
        got = planarity_of(pts)
        check(f'{name} -> NaN rather than a fake 100%',
              all(np.isnan(v) for v in got), str(got))

    # 5. plane_project agrees with planarity_of's rmsd -- the claim that lets
    #    visualize_geometry.py drop its own copy.
    worst = 0.0
    for trial in range(50):
        n = int(rng.integers(3, 40))
        pts = rng.normal(size=(n, 3)) * np.array([3.0, 2.0, 0.3])
        _, oop = plane_project(pts)
        worst = max(worst, abs(oop - planarity_of(pts)[2]))
    check('plane_project out-of-plane RMS == planarity rmsd', worst < 1e-12,
          f'worst abs difference {worst:.2e}')

    # 6. The SMARTERS threshold ladder is the documented one.
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
