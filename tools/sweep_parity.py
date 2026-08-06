#!/usr/bin/env python3
"""
Faithfulness check for the native Lean hyperparameter-sweep math (Puffer/RL/Sweep.lean) vs PufferLib's
`pufferlib.sweep`.  Compares the DETERMINISTIC transforms (a stochastic optimizer is otherwise
unverifiable): the Space normalize/unnormalize/scale resolution, and the Hyperparameters flat-key order,
min/max bounds, search scales, and to_dict/from_dict — at fixed inputs.

Runs `.lake/build/bin/puffer verify-sweep-spaces` (which prints each quantity as exact float BITS), computes
the same reference from `pufferlib.sweep` (+ our config/default.ini, which the Lean side also reads), and
reports max |Δ|.  Run with an interpreter that has pufferlib installed, e.g.:
    /home/st/src/PufferLib/.venv/bin/python tools/sweep_parity.py
"""
import struct, subprocess, sys, math, configparser, ast, os

def bits_to_float(b):
    return struct.unpack('<d', struct.pack('<Q', int(b) & 0xFFFFFFFFFFFFFFFF))[0]

def run_lean():
    out = subprocess.run(['.lake/build/bin/puffer', 'verify-sweep-spaces'],
                         capture_output=True, text=True, env={**os.environ, 'LD_LIBRARY_PATH': ''})
    if out.returncode != 0:
        print('lean binary failed:\n', out.stderr); sys.exit(1)
    vals, keys = {}, {}
    for line in out.stdout.splitlines():
        p = line.split()
        if not p: continue
        tag = p[0]
        if tag == 'KEY':      keys[int(p[1])] = p[2]
        elif tag == 'NUM':    vals['NUM'] = int(p[1])
        elif tag in ('SCALE',):        vals[f'{tag} {p[1]}'] = bits_to_float(p[2])
        elif tag in ('NORM','UNNORM'): vals[f'{tag} {p[1]} {p[2]}'] = bits_to_float(p[3])
        elif tag in ('NMIN','NMAX','SCL','TODICT','FROMDICT'): vals[f'{tag} {int(p[1])}'] = bits_to_float(p[2])
    return vals, keys

def load_sweep_config(path):
    """Replicate pufferl.load_config's sweep extraction on OUR config/default.ini (what Lean reads)."""
    p = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))  # Lean's parseIni strips inline # too
    p.read(path)
    sweep = {}
    for section in p.sections():
        if section != 'sweep' and not section.startswith('sweep.'):
            continue
        for key in p[section]:
            try:    value = ast.literal_eval(p[section][key])
            except: value = p[section][key]
            path_keys = [key] if section == 'sweep' else section.split('.')[1:] + [key]
            d = sweep
            for sk in path_keys[:-1]:
                d = d.setdefault(sk, {})
            d[path_keys[-1]] = value
    return sweep

def main():
    import pufferlib.sweep as S
    lean, lkeys = run_lean()

    max_abs = 0.0   # for normalized-space quantities (all O(1))
    max_rel = 0.0   # for unnormalized values (huge dynamic range)
    fails = []
    def cmp_abs(label, ref):
        nonlocal max_abs
        if label not in lean: fails.append(f'missing {label}'); return
        d = abs(lean[label] - ref);
        globals()  # noqa
        max_abs = max(max_abs, d)
        if d > 1e-9: fails.append(f'{label}: lean={lean[label]!r} ref={ref!r} |Δ|={d:.3e}')
    def cmp_rel(label, ref):
        nonlocal max_rel
        if label not in lean: fails.append(f'missing {label}'); return
        d = abs(lean[label] - ref)
        r = d / (abs(ref) + 1e-300)
        max_rel = max(max_rel, r)
        if r > 1e-9 and d > 1e-9: fails.append(f'{label}: lean={lean[label]!r} ref={ref!r} rel={r:.3e}')

    # (1) Space battery — must match Lean's verifySweepSpaces exactly.
    battery = [
        ('uniform',      S.Linear(0.1, 5.0, 0.5)),
        ('int_uniform',  S.Linear(1, 8, 'auto', is_integer=True)),
        ('uniform_pow2', S.Pow2(32, 1024, 'auto', is_integer=True)),
        ('log_normal',   S.Log(1e-5, 0.1, 0.5)),
        ('log_time',     S.Log(3e7, 1e11, 'time')),
        ('logit_normal', S.Logit(0.8, 0.9999, 'auto')),
    ]
    norm_pts = [-1.0, -0.3, 0.4, 1.0]
    for nm, sp in battery:
        cmp_abs(f'SCALE {nm}', sp.scale)
        mids = [sp.min, sp.max, (sp.min + sp.max) / 2.0]
        for k in range(3):  cmp_abs(f'NORM {nm} {k}', sp.normalize(mids[k]))
        for k in range(4):  cmp_rel(f'UNNORM {nm} {k}', sp.unnormalize(norm_pts[k]))

    # (2) Full default.ini Hyperparameters.
    sweep_cfg = load_sweep_config('config/default.ini')
    hp = S.Hyperparameters(sweep_cfg, verbose=False)
    key_ok = (lean.get('NUM') == hp.num)
    ref_keys = list(hp.flat_spaces.keys())
    if lean.get('NUM') != hp.num:
        fails.append(f'NUM: lean={lean.get("NUM")} ref={hp.num}')
    for i, (k, sp) in enumerate(hp.flat_spaces.items()):
        if lkeys.get(i) != k:
            key_ok = False; fails.append(f'KEY {i}: lean={lkeys.get(i)} ref={k}')
        cmp_abs(f'NMIN {i}', hp.min_bounds[i])
        cmp_abs(f'NMAX {i}', hp.max_bounds[i])
        cmp_abs(f'SCL {i}',  hp.search_scales[i])
        s = math.sin(i)
        v = sp.unnormalize(s)
        cmp_rel(f'TODICT {i}', v)
        cmp_abs(f'FROMDICT {i}', sp.normalize(v))

    print(f'params (NUM):        lean={lean.get("NUM")}  pufferlib={hp.num}   flat-key order match: {key_ok}')
    print(f'flat keys:           {ref_keys}')
    print(f'max |Δ| (norm-space, scales, bounds, from_dict): {max_abs:.3e}')
    print(f'max relΔ (unnormalize / to_dict values):         {max_rel:.3e}')
    if fails:
        print(f'\nFAIL — {len(fails)} discrepancy(ies) over the 1e-9 bar:')
        for f in fails[:30]: print('  ', f)
        sys.exit(1)
    print('\nPASS — native Space/Hyperparameters math matches pufferlib.sweep to < 1e-9.')

if __name__ == '__main__':
    main()
