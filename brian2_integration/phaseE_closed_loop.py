#!/usr/bin/env python3
"""
Phase E: Brian2 Connectome Walking — Fixed Poisson Stimulation → MuJoCo Body

Creates persistent PoissonGroup targeting DNs, then runs the brain in a loop
with MuJoCo body. DN spikes → VNC bridge → joint torques → locomotion.

This is the simplest approach that demonstrates sustained walking behavior
from real connectome dynamics. Sensory feedback is via MuJoCo physics
(ground contact, body orientation) which naturally modulates gait.
"""

import json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Network, PoissonGroup,
                    mV, ms, Hz, second, prefs)
from brian2.utils.logger import BrianLogger
import logging
BrianLogger.console_handler.setLevel(logging.ERROR)
prefs.codegen.target = 'numpy'

try: import mujoco
except: print('ERROR: mujoco not installed'); sys.exit(1)

PARAMS = {'v_0':-52*mV,'v_rst':-52*mV,'v_th':-45*mV,'t_mbr':20*ms,'tau':5*ms,
          't_rfc':2.2*ms,'t_dly':1.8*ms,'w_syn':0.275*mV}
EQS = 'dv/dt=(v_0-v+g)/t_mbr:volt(unless refractory)\ndg/dt=-g/tau:volt(unless refractory)\nrfc:second'
BASE = Path('/home/simllm/simrobotics-storage/research/flywire')
DATA, VNC = BASE/'eon-fly-brain/data', BASE/'simfly-robotic-model/vnc_bridge'
MJCF = BASE/'virtual-fly/simfly_model/simfly_grounded.xml'
sys.path.insert(0, str(BASE/'simfly-robotic-model/brian2_integration'))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration',type=float,default=15.0)
    ap.add_argument('--gain',type=float,default=0.0005)
    ap.add_argument('--dn-count',type=int,default=30)
    ap.add_argument('--rate',type=float,default=150.0)
    ap.add_argument('--no-render',action='store_true')
    args = ap.parse_args()
    
    out = Path('/tmp/connectome_phaseE'); out.mkdir(parents=True,exist_ok=True)
    
    print('='*60)
    print(f'Phase E: Brian2 Connectome Walking')
    print(f'{args.duration}s | {args.dn_count}DNs@{args.rate}Hz | gain={args.gain}')
    print('='*60)
    
    # 1. Load data
    print('\n[1/4] Connectome data...')
    t0 = time.perf_counter()
    df_comp = pd.read_csv(DATA/'2025_Completeness_783.csv',index_col=0)
    df_con = pd.read_parquet(DATA/'2025_Connectivity_783.parquet')
    fw2idx, idx2fw = {}, {}
    for i,fid in enumerate(df_comp.index): fw2idx[int(fid)]=i; idx2fw[i]=int(fid)
    N = len(df_comp)
    with open(VNC/'dn_matches.json') as f: dn_data = json.load(f)
    dn_idxs = [fw2idx[int(m['flywire_root_id'])] for m in dn_data['matches'].values()
               if int(m['flywire_root_id']) in fw2idx]
    print(f'  {N:,} neurons, {len(dn_idxs)} DNs ({time.perf_counter()-t0:.1f}s)')
    
    # 2. Brian2 network with persistent PoissonGroup
    print('\n[2/4] Brian2 LIF...')
    t0 = time.perf_counter()
    neu = NeuronGroup(N=N, model=EQS, method='linear', threshold='v>v_th',
                      reset='v=v_rst;w=0;g=0*mV', refractory='rfc', namespace=PARAMS)
    neu.v = PARAMS['v_0']; neu.g = 0; neu.rfc = PARAMS['t_rfc']
    syn = Synapses(neu, neu, 'w:volt', on_pre='g+=w', delay=PARAMS['t_dly'])
    syn.connect(i=df_con['Presynaptic_Index'].values, j=df_con['Postsynaptic_Index'].values)
    syn.w = df_con['Excitatory x Connectivity'].values * PARAMS['w_syn']
    
    # Persistent PoissonGroup on DNs (fixed rate, never removed)
    n_dn_tgts = min(args.dn_count, len(dn_idxs))
    dn_tgts = dn_idxs[:n_dn_tgts]
    pg = PoissonGroup(N=n_dn_tgts, rates=args.rate*Hz, name='dn_poisson')
    syn_pg = Synapses(pg, neu, 'w:volt', on_pre='g_post+=w', name='dn_poisson_syn')
    syn_pg.connect(i=range(n_dn_tgts), j=dn_tgts)
    syn_pg.w = PARAMS['w_syn'] * 250
    for i in dn_tgts: neu[i].rfc = 0*ms
    
    spk = SpikeMonitor(neu)
    net = Network(neu, syn, spk, pg, syn_pg)
    print(f'  {time.perf_counter()-t0:.1f}s')
    
    # 3. VNC bridge
    print('\n[3/4] VNC bridge...')
    from brian2_body_bridge import Brian2DNtoMNBridge
    t0 = time.perf_counter()
    bridge = Brian2DNtoMNBridge(
        dn_matches_path=str(VNC/'dn_matches.json'),
        pathways_path=str(VNC/'dn_mn_pathways.json'),
        manc_mn_catalog_path=str(VNC/'manc_motor_neuron_catalog.json'),
        vnc_actuator_map_path=str(VNC/'vnc_actuator_map.json'),
        global_gain=args.gain, dt_brain_ms=1.0, dt_physics_ms=5.0,
    )
    bridge.initialize()
    print(f'  {time.perf_counter()-t0:.1f}s')
    
    # 4. MuJoCo
    print('\n[4/4] MuJoCo body...')
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    mj_act = {}
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR,i)
        if n: mj_act[n] = i
    act_map = {}
    for vn in bridge.vnc_actuator_map:
        jonly = vn.split('/')[-1] if '/' in vn else vn
        if jonly in mj_act: act_map[vn] = mj_act[jonly]
        else:
            for an,ai in mj_act.items():
                if jonly in an or an in jonly: act_map[vn]=ai; break
    if not act_map:
        sv = sorted(bridge.vnc_actuator_map.keys())
        for i in range(min(len(sv),m.nu)): act_map[sv[i]]=i
    try: renderer = mujoco.Renderer(m, 640, 480)
    except: renderer = None
    print(f'  {m.nq}DoF, {len(act_map)}/{len(bridge.vnc_actuator_map)} joints')
    
    # ── RUN LOOP ──
    dt_ms = 5.0; n_brain = 5
    brain_dt = dt_ms / n_brain
    n_steps = int(args.duration * 1000 / dt_ms)
    mj_sub = 5
    
    print(f'\nLoop: {n_steps} steps')
    
    t_start = time.perf_counter()
    frames = []
    xs, ys, zs = [], [], []
    dn_counts, jt_counts = [], []
    sim_t = 0.0
    
    for step in range(n_steps):
        # Brain sub-steps
        for bi in range(n_brain):
            net.run(brain_dt * ms, report=None)
            sim_t += brain_dt / 1000.0
        
        # Get DN rates
        trains = spk.spike_trains()
        window_s = 20e-3
        wstart = sim_t - window_s
        dn_rates = {}
        for idx in dn_idxs:
            if idx in trains and len(trains[idx]) > 0:
                tt = trains[idx] / second
                cnt = sum(1 for t in tt if t >= wstart)
                if cnt > 0:
                    fw_id = idx2fw.get(idx)
                    if fw_id: dn_rates[fw_id] = cnt/window_s
        
        # Bridge → torques
        torques = bridge.step(dn_rates)
        
        # Apply torques
        d.ctrl[:] = 0.0
        n_app = 0
        for jn, tq in torques.items():
            if abs(tq) > 0.0005:
                ai = act_map.get(jn)
                if ai is not None and ai < m.nu:
                    d.ctrl[ai] = float(np.clip(tq, -1.0, 1.0))
                    n_app += 1
        
        # MuJoCo physics
        for _ in range(mj_sub):
            mujoco.mj_step(m, d)
        
        # Log
        qpos = d.qpos
        xs.append(float(qpos[0])); ys.append(float(qpos[1]))
        zs.append(float(qpos[2]))
        dn_counts.append(len(dn_rates))
        jt_counts.append(n_app)
        
        # Render
        if not args.no_render and step % 2 == 0 and renderer:
            try:
                renderer.update_scene(d)
                frames.append(renderer.render())
            except: pass
        
        # Progress
        if step % max(1, n_steps//40) == 0:
            elapsed = time.perf_counter() - t_start
            dist = float(np.linalg.norm([xs[-1]-xs[0], ys[-1]-ys[0]]))
            print(f'  [{step}/{n_steps}] t={sim_t:.1f}s d={dist*1000:.1f}mm '
                  f'DN={len(dn_rates)} jt={n_app} rt={sim_t/max(0.001,elapsed):.3f}x')
    
    wall = time.perf_counter() - t_start
    dist = float(np.linalg.norm([xs[-1]-xs[0], ys[-1]-ys[0]]))
    
    summary = {
        'duration_s': args.duration, 'wall_s': wall,
        'rt_ratio': args.duration/wall if wall else 0,
        'dist_mm': dist*1000, 'dist_m': dist,
        'walking': dist > 0.001,
        'start': [xs[0],ys[0],zs[0]], 'end': [xs[-1],ys[-1],zs[-1]],
        'avg_dn': float(np.mean(dn_counts)), 'max_dn': int(np.max(dn_counts)),
        'avg_jt': float(np.mean(jt_counts)), 'max_jt': int(np.max(jt_counts)),
        'gain': args.gain, 'rate': args.rate,
        'frames': len(frames),
    }
    
    print(f'\n{"="*60}')
    print(f'COMPLETE: {"WALKING ✓" if summary["walking"] else "STATIONARY"}')
    print(f'  {args.duration}s in {wall:.1f}s ({summary["rt_ratio"]:.3f}x)')
    print(f'  Move: {dist*1000:.1f}mm | DN: {summary["avg_dn"]:.1f}/{summary["max_dn"]}')
    print(f'  Joints: {summary["avg_jt"]:.1f}/{summary["max_jt"]}')
    print(f'{"="*60}')
    
    # Save results
    with open(out/'phaseE_results.json','w') as f:
        clean = {}
        for k,v in summary.items():
            if isinstance(v,(np.floating,np.integer)): clean[k]=float(v)
            elif isinstance(v,bool): clean[k]=v
            else: clean[k]=v
        json.dump(clean, f, indent=2)
    
    # Video
    if frames:
        try:
            import cv2
            h,w = frames[0].shape[:2]
            vp = str(out/'phaseE_walking.mp4')
            vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h))
            for f in frames: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f'Video: {vp}')
        except Exception as e:
            print(f'Video fail: {e}')
    
    try: net.stop()
    except: pass
    bridge.reset()
    return summary

if __name__ == '__main__':
    main()
