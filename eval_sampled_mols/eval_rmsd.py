"""Compare saved generated coordinates with their target coordinates."""

import argparse
import json
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.rmsd import hungarian_rmsd_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir', required=True,
                        help='checkpoint folder containing the generated and target molecule lists')
    parser.add_argument('--predictions', default='sample_rdmols_3d.pkl',
                        help='generated molecule filename inside --samples-dir')
    parser.add_argument('--max-distance', type=float, default=5.0,
                        help='matching cutoff including atom-type penalties, in angstroms')
    parser.add_argument('--min-atoms', type=int, default=3,
                        help='minimum number of mapped atoms')
    parser.add_argument('--out', default=None,
                        help='JSON path (default: rmsd.json inside --samples-dir)')
    args = parser.parse_args()
    if not math.isfinite(args.max_distance) or args.max_distance <= 0 or args.min_atoms < 1:
        parser.error('--max-distance must be finite and positive; --min-atoms must be positive')
    lists = []
    for name in ('groundtruth_rdmols.pkl', args.predictions):
        path = os.path.join(args.samples_dir, name)
        if not os.path.isfile(path):
            parser.error(f'missing molecule file: {path}')
        with open(path, 'rb') as f:
            lists.append(pickle.load(f))
    truth, predictions = lists
    if not len(truth) or len(truth) != len(predictions):
        parser.error('target and prediction lists must be nonempty and have equal lengths')
    rmsds, rate, mean, atom_accuracy = hungarian_rmsd_batch(
        truth, predictions, max_distance=args.max_distance, min_atoms=args.min_atoms)
    result = dict(n_pairs=len(truth), rmsd=rmsds, success_rate=rate,
                  mean_rmsd=mean, mean_atom_type_accuracy=atom_accuracy,
                  parameters=dict(max_distance=args.max_distance, min_atoms=args.min_atoms))
    output = args.out or os.path.join(args.samples_dir, 'rmsd.json')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
