#!/usr/bin/env python3
"""Turn a SLURM training log into a CSV of training dynamics.

    python3 tools/parse_train_log.py slurm/logs/ters-train-925143.out -o curve.csv
    python3 tools/parse_train_log.py slurm/logs/ters-train-*.out --plot curve.png

Works on a job that is still running -- just re-run it to pick up new steps.

Columns: step, loss, elapsed_min, steps_per_s, mol_per_s, loss_ema

Why this exists: run_lib.py:243 is the ONLY per-step record this repo emits, and it
goes to stdout, not to an event file (the tensorboard import at run_lib.py:8 is
commented out). So the log IS the dataset. Resumes are handled by keeping the last
occurrence of each step, since a restart replays step numbers.
"""
import argparse, csv, re, sys
from datetime import datetime, timedelta

# I0828 13:34:59.822825 70368825917728 run_lib.py:243] step: 0, training_loss: 6.79754e+01
LINE = re.compile(
    r'^[A-Z](?P<md>\d{4})\s+(?P<hms>\d{2}:\d{2}:\d{2}\.\d+).*?'
    r'step:\s*(?P<step>\d+),\s*training_loss:\s*(?P<loss>[-\d.eE+]+)'
)
# batch_size lands in the config dump the run prints at startup
BATCH = re.compile(r'^\s*batch_size:\s*(\d+)\s*$')


def parse(paths):
    rows, batch = {}, None
    for path in paths:
        with open(path, errors='replace') as fh:
            for line in fh:
                if batch is None:
                    b = BATCH.match(line)
                    if b:
                        batch = int(b.group(1))
                m = LINE.match(line)
                if not m:
                    continue
                t = datetime.strptime(f"{m['md']} {m['hms']}", "%m%d %H:%M:%S.%f")
                # last write wins, so a resumed run overwrites replayed steps
                rows[int(m['step'])] = (t, float(m['loss']))
    return [(s, *rows[s]) for s in sorted(rows)], batch


def enrich(recs, batch, ema_alpha=0.1):
    out, ema = [], None
    t0 = recs[0][1] if recs else None
    for i, (step, t, loss) in enumerate(recs):
        # a resume can rewind the clock or cross midnight; treat those as gaps
        elapsed = (t - t0).total_seconds()
        if elapsed < 0:
            elapsed += timedelta(days=1).total_seconds()
        sps = mps = ''
        if i:
            ps, pt, _ = recs[i - 1]
            dt = (t - pt).total_seconds()
            ds = step - ps
            if dt > 0 and ds > 0:
                sps = round(ds / dt, 4)
                if batch:
                    mps = round(ds * batch / dt, 2)
        ema = loss if ema is None else ema_alpha * loss + (1 - ema_alpha) * ema
        out.append(dict(step=step, loss=loss, elapsed_min=round(elapsed / 60, 3),
                        steps_per_s=sps, mol_per_s=mps, loss_ema=round(ema, 6)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('-o', '--out', default='-', help='CSV path, or - for stdout')
    ap.add_argument('--plot', metavar='PNG', help='also write a loss curve here')
    a = ap.parse_args()

    recs, batch = parse(a.logs)
    if not recs:
        sys.exit('no "step: N, training_loss: X" lines found -- wrong file?')
    rows = enrich(recs, batch)

    fh = sys.stdout if a.out == '-' else open(a.out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    if fh is not sys.stdout:
        fh.close()

    last = rows[-1]
    print(f"{len(rows)} points | step {rows[0]['step']}..{last['step']} "
          f"| loss {rows[0]['loss']:.4g} -> {last['loss']:.4g} "
          f"(ema {last['loss_ema']:.4g}) | {last['elapsed_min']:.1f} min"
          + (f" | batch {batch}" if batch else ""), file=sys.stderr)

    if a.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit('--plot needs matplotlib; the CSV was still written')
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot([r['step'] for r in rows], [r['loss'] for r in rows],
                lw=.8, alpha=.35, label='loss')
        ax.plot([r['step'] for r in rows], [r['loss_ema'] for r in rows],
                lw=1.8, label='EMA(0.1)')
        ax.set_xlabel('step'); ax.set_ylabel('training loss')
        ax.set_title('TERS DiffSpectra training'); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print(f'wrote {a.plot}', file=sys.stderr)


if __name__ == '__main__':
    main()
