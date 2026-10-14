"""
Step-API fly-brain controller for the drone project.

Wraps the benchmarked PyTorch model (run_pytorch.TorchModel) so it can be
stepped in a closed loop:
  - time-varying sensory input rates (encoder -> `rates` tensor each period)
  - persistent activation (behavior channel, optogenetics semantics)
  - silencing (zero all synapses to AND from the given neurons, per README)
  - firing-rate readout of an arbitrary neuron subset (e.g. DNs)
  - no per-step GPU->CPU sync (the benchmark's spike recording forced one
    every step; we only transfer the small readout at control-period ends)

Numerical fidelity is inherited: we import the exact classes and constants
from the benchmark runner, nothing is reimplemented.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
from run_pytorch import (  # noqa: E402
    DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights,
)


class FlyBrain:
    """Fly connectome as a controller: rates in -> firing rates out."""

    def __init__(self, comp_path, conn_path, wt_dir,
                 device=None, n_replicates=1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.comp_path, self.conn_path, self.wt_dir = (
            str(comp_path), str(conn_path), str(wt_dir))
        self.flyid2i, self.i2flyid = get_hash_tables(self.comp_path)
        self.B = n_replicates
        self.n = len(self.flyid2i)
        self._driven = []     # flywire ids fed by the encoder
        self._silenced = []   # flywire ids with synapses zeroed
        self._act_rate = {}   # index -> persistent activation Hz
        self._readout = torch.arange(self.n, device=self.device)
        self._rebuild()

    # ---------- setup ----------
    def _rebuild(self):
        weights = get_weights(self.conn_path, self.comp_path,
                              self.wt_dir, csr=True).to(self.device)
        if self._silenced:
            idxs = [self.flyid2i[f] for f in self._silenced]
            weights = self._zero_neurons(weights, idxs)
        # driven neurons get refractory disabled (paper's `neu[i].rfc = 0`)
        exc = [self.flyid2i[f] for f in self._driven]
        self.model = TorchModel(self.B, self.n, DT, MODEL_PARAMS, weights,
                                exc_indices=exc, device=self.device)
        self.state = self.model.state_init()
        self.rates = torch.zeros(self.B, self.n, device=self.device)
        self.t_sim = 0.0

    @staticmethod
    def _zero_neurons(weights, idxs):
        """CSR copy with all synapses onto and from `idxs` zeroed.
        Note: paper model.py zeroes outgoing only; README documents
        to-and-from. We follow the README (full lesion)."""
        crow = weights.crow_indices().clone()
        col = weights.col_indices().clone()
        val = weights.values().clone()
        kill = torch.isin(col, torch.as_tensor(idxs, device=val.device,
                                               dtype=torch.long))  # outgoing
        for k in idxs:                                        # incoming
            kill[crow[k]:crow[k + 1]] = True
        val = torch.where(kill, torch.zeros_like(val), val)
        return torch.sparse_csr_tensor(crow, col, val, weights.shape,
                                       device=weights.device)

    # ---------- command interface ----------
    def set_driven(self, flywire_ids):
        """Sensory entry neurons (encoder inputs). Rebuilds the model."""
        self._driven = list(flywire_ids)
        self._rebuild()

    def set_silenced(self, flywire_ids):
        self._silenced = list(flywire_ids)
        self._rebuild()

    def set_readout(self, flywire_ids):
        self._readout = torch.tensor([self.flyid2i[f] for f in flywire_ids],
                                     device=self.device)

    def activate(self, flywire_ids, rate_hz):
        for f in flywire_ids:
            self._act_rate[self.flyid2i[f]] = rate_hz

    def clear_activation(self):
        self._act_rate.clear()

    # ---------- stepping ----------
    def _set_rates(self, input_rates):
        self.rates.zero_()
        for i, r in self._act_rate.items():
            self.rates[:, i] = r
        if input_rates:
            for f, r in input_rates.items():
                self.rates[:, self.flyid2i[f]] = r

    def _run(self, duration_s):
        ticks = int(round(duration_s * 1000.0 / DT))
        counts = torch.zeros(self.n, device=self.device)
        with torch.no_grad():
            for _ in range(ticks):
                self.state = self.model(self.rates, *self.state)
                counts += self.state[2].sum(0)   # spikes summed over batch
        self.t_sim += duration_s
        return counts

    def step(self, duration_s, input_rates=None):
        """One control period. Returns readout firing rates (Hz, np array)."""
        self._set_rates(input_rates)
        counts = self._run(duration_s)
        return (counts[self._readout] / (self.B * duration_s)).cpu().numpy()

    def measure(self, duration_s, input_rates=None):
        """Fresh-state trial (like a paper trial). Returns all-N rates (Hz)."""
        self.state = self.model.state_init()
        self._set_rates(input_rates)
        counts = self._run(duration_s)
        return (counts / (self.B * duration_s)).cpu().numpy()

    def probe(self, stim_rates, duration_s=1.0, top=25):
        """Differential probing: baseline vs stimulus, most-activated neurons.
        Excludes the driven neurons themselves (they echo the input).
        Returns list of (flywire_id, base_hz, stim_hz, delta_hz)."""
        driven = set(self._driven)
        base = self.measure(duration_s)
        stim = self.measure(duration_s, stim_rates)
        delta = stim - base
        order = np.argsort(-delta)
        rows = []
        for i in order:
            fid = self.i2flyid[int(i)]
            if fid in driven:
                continue
            rows.append((fid, float(base[i]), float(stim[i]), float(delta[i])))
            if len(rows) >= top:
                break
        return rows


# ---------- helpers ----------

def ids_by_type(comp_path, pattern, verbose=True):
    """Flywire IDs whose cell-type string matches `pattern` (regex fullmatch).
    Auto-detects the column holding type names. Returns [] and prints the
    available columns if nothing matches."""
    df = pd.read_csv(comp_path, index_col=0)
    for c in df.columns:
        if df[c].dtype == object:
            m = df[c].astype(str).str.fullmatch(pattern, na=False)
            if m.any():
                ids = df.index[m].tolist()
                if verbose:
                    print(f"[ids_by_type] '{c}' ~ '{pattern}': {len(ids)} neurons")
                return ids
    print(f"[ids_by_type] no match for '{pattern}'. Columns: {list(df.columns)}")
    return []


class LoomingEncoder:
    """Range + closing speed -> LC4/LC16 population rate (Hz).
    Rate rises as time-to-contact shrinks; saturates at r_max."""

    def __init__(self, r_max=200.0, ttc_scale=0.5):
        self.r_max, self.ttc_scale = r_max, ttc_scale

    def __call__(self, z, zdot):
        ttc = z / max(zdot, 1e-6)          # time to contact, seconds
        return self.r_max / (1.0 + ttc / self.ttc_scale)


class StopDecoder:
    """Readout population rate -> commanded-velocity scale in [0, 1]."""

    def __init__(self, threshold_hz=30.0):
        self.thr = threshold_hz

    def __call__(self, rate_hz):
        return float(np.clip(1.0 - rate_hz / self.thr, 0.0, 1.0))


class MockBrain:
    """Instant stand-in with FlyBrain's step() interface. Readout echoes the
    max input rate — used to test the control-loop plumbing in seconds."""

    def set_driven(self, ids): pass
    def set_readout(self, ids): pass

    def step(self, duration_s, input_rates=None):
        r = max(input_rates.values()) if input_rates else 0.0
        return np.array([r])

