#!/usr/bin/env python3
"""1-D stopping demo: drone flies toward a wall; the fly brain must stop it.

Phase A (first run per entry): probe — stimulate the entry neurons, rank all
neurons by firing-rate increase, cache the top-25 as the readout
(data/drone/readout_<entry>.json).

Phase B: closed loop. The looming encoder drives the entry neurons with
rate ~ 1/TTC; the decoder scales commanded velocity by the readout
population's rate.

Run from repo root:
    python3 code/drone/sim1d.py --mock                  # plumbing test, seconds
    python3 code/drone/sim1d.py --entry sugar           # real brain, ~30-40 min
    python3 code/drone/sim1d.py --entry sugar --readout p9 --replicates 4
    python3 code/drone/sim1d.py --entry p9 --probe      # map P9's downstream
    python3 code/drone/sim1d.py --entry loom           # once annotations file exists
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from brain_controller import FlyBrain, MockBrain, ids_by_type, \
                             LoomingEncoder, StopDecoder

ROOT = Path(__file__).resolve().parents[2]
COMP = ROOT / 'data/2025_Completeness_783.csv'
CONN = ROOT / 'data/2025_Connectivity_783.parquet'
WTDIR = ROOT / 'data'
DRONE_DIR = ROOT / 'data/drone'


def get_entry_ids(entry):
    """Resolve --entry to (flywire_ids, label, probe_rate_hz)."""
    if entry == 'loom':
        ids = ids_by_type(COMP, r'LC4.*') + ids_by_type(COMP, r'LC16.*')
        if not ids:
            raise SystemExit(
                'LC4/LC16 not found. Download the FlyWire 783 cell-type table '
                'from codex.flywire.ai to data/2025_Annotations_783.csv, '
                'or use --entry sugar / --entry p9.')
        return ids, 'LC4/LC16 (looming)', 150.0
    from benchmark import EXPERIMENTS, get_experiment
    if entry not in EXPERIMENTS:
        raise SystemExit(f"Unknown entry '{entry}'. "
                         f"Available: {list(EXPERIMENTS)} plus 'loom'")
    exp = get_experiment(entry)
    return list(exp['neu_exc']), exp['name'], float(exp.get('stim_rate', 200.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--probe', action='store_true',
                    help='force re-run of the probing phase')
    ap.add_argument('--entry', default='sugar',
                    help="'loom' (needs annotations file) or an EXPERIMENTS "
                         "key: sugar, p9")
    ap.add_argument('--readout', choices=['auto', 'p9'], default='auto',
                    help='auto: probe-derived top-25; p9: forward-walking pair')
    ap.add_argument('--v0', type=float, default=1.0, help='cruise speed m/s')
    ap.add_argument('--z0', type=float, default=5.0, help='start distance m')
    ap.add_argument('--t-max', type=float, default=6.0)
    ap.add_argument('--dt-ctrl', type=float, default=0.05)
    ap.add_argument('--thr', type=float, default=30.0,
                    help='decoder threshold Hz')
    ap.add_argument('--replicates', type=int, default=1)
    args = ap.parse_args()

    DRONE_DIR.mkdir(parents=True, exist_ok=True)
    readout_json = DRONE_DIR / f'readout_{args.entry}.json'
    log_path = DRONE_DIR / f'sim1d_{args.entry}_log.csv'

    # ---- entry neurons ----
    entry_ids, entry_label, probe_rate = get_entry_ids(args.entry)
    print(f'>>> Entry neurons: {entry_label} ({len(entry_ids)} cells)')

    # ---- brain ----
    if args.mock:
        brain = MockBrain()
    else:
        brain = FlyBrain(COMP, CONN, WTDIR, n_replicates=args.replicates)
        brain.set_driven(entry_ids)

    # ---- readout ----
    if args.readout == 'p9':
        from benchmark import EXPERIMENTS
        readout_ids = list(EXPERIMENTS['p9']['neu_exc'])
        brain.set_readout(readout_ids)
        print('>>> Readout: P9 pair (forward-walking command neurons)')
    elif args.mock:
        readout_ids = entry_ids
    elif args.probe or not readout_json.is_file():
        print(f'>>> Probing: {entry_label} @{probe_rate:.0f} Hz vs baseline '
              f'(2 fresh 1-s trials)...')
        rows = brain.probe({f: probe_rate for f in entry_ids},
                           duration_s=1.0, top=25)
        print(f'{"flywire_id":>22} {"base_hz":>8} {"stim_hz":>8} {"delta":>8}')
        for fid, b, s, d in rows:
            print(f'{fid:>22} {b:8.1f} {s:8.1f} {d:8.1f}')
        readout_ids = [r[0] for r in rows]
        json.dump({'ids': [str(i) for i in readout_ids],
                   'rows': [[str(r[0]), r[1], r[2], r[3]] for r in rows]},
                  open(readout_json, 'w'), indent=1)
        brain.set_readout(readout_ids)
        print(f'>>> Cached {len(readout_ids)} readout neurons to {readout_json.name}')
    else:
        readout_ids = [int(i) for i in json.load(open(readout_json))['ids']]
        brain.set_readout(readout_ids)
        print(f'>>> Readout: {len(readout_ids)} cached neurons '
              f'({readout_json.name})')

    enc = LoomingEncoder()
    dec = StopDecoder(threshold_hz=args.thr)

    # ---- closed loop ----
    z, v = args.z0, 0.0
    tau_v = 0.3                      # velocity tracking lag, s
    dt = args.dt_ctrl
    log = [['t', 'z', 'v', 'in_rate_hz', 'readout_hz', 'v_scale']]
    print(f'>>> Mission: approach wall from {args.z0} m at {args.v0} m/s '
          f'(entry: {entry_label})')

    for k in range(int(args.t_max / dt)):
        t = k * dt
        rate = enc(z, max(v, 1e-6))                 # sensor -> entry-neuron rate
        ro_hz = float(np.mean(brain.step(           # brain step, mean readout Hz
            dt, {f: rate for f in entry_ids})))
        scale = dec(ro_hz)                          # decoder -> setpoint
        v_cmd = args.v0 * scale
        v += (v_cmd - v) / tau_v * dt               # 1st-order velocity lag
        z -= v * dt
        log.append([round(t, 3), round(z, 4), round(v, 4),
                    round(rate, 1), round(ro_hz, 1), round(scale, 3)])

        if z <= 0.02:
            print(f'>>> CRASH into wall at t={t:.2f}s')
            break
        if v < 0.02:
            print(f'>>> STOPPED at z={z:.3f} m, t={t:.2f}s (approach '
                  f'speed {args.v0} m/s)')
            break
    else:
        print(f'>>> TIMEOUT: z={z:.3f} m, v={v:.3f} m/s')

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerows(log)
    print(f'>>> Trajectory: {log_path}')


if __name__ == '__main__':
    main()
