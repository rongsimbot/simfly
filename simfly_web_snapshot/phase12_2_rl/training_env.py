#!/usr/bin/env python3
"""
Per-Joint RL Training Environment — Gymnasium-compatible wrapper for Brain2 API.

RL does NOT control the fly — it only tunes the motor decoder interface.
The connectome provides ALL neural signals.

State:  36 torque values + food_distance + z_height + fired_neurons + 
        dn_matches + mns_activated + active_joints + wall_distance -> 43 dims
Action: 36 per-joint multipliers in [0.5, 2.0] (continuous)
Reward: torque_CV * 10 + food_distance_reduction * 5 - saturation_penalty
"""
import time
import json
import math
import numpy as np
import urllib.request
import urllib.error


class Brain2Env:
    """Gymnasium-style environment wrapping Brain2 REST API."""
    
    JOINT_NAMES = [
        "coxa_T1_left", "coxa_T1_right", "coxa_T2_left", "coxa_T2_right",
        "coxa_T3_left", "coxa_T3_right",
        "coxa_abduct_T1_left", "coxa_abduct_T1_right", "coxa_abduct_T2_left",
        "coxa_abduct_T2_right", "coxa_abduct_T3_left", "coxa_abduct_T3_right",
        "coxa_twist_T1_left", "coxa_twist_T1_right", "coxa_twist_T2_left",
        "coxa_twist_T2_right", "coxa_twist_T3_left", "coxa_twist_T3_right",
        "femur_T1_left", "femur_T1_right", "femur_T2_left", "femur_T2_right",
        "femur_T3_left", "femur_T3_right",
        "femur_twist_T1_left", "femur_twist_T1_right", "femur_twist_T2_left",
        "femur_twist_T2_right", "femur_twist_T3_left", "femur_twist_T3_right",
        "tibia_T1_left", "tibia_T1_right", "tibia_T2_left", "tibia_T2_right",
        "tibia_T3_left", "tibia_T3_right",
    ]
    
    N_JOINTS = len(JOINT_NAMES)
    STATE_DIM = N_JOINTS + 7
    
    def __init__(self, base_url='http://192.168.1.199:8080',
                 global_gain=0.001, tau_decay=100.0,
                 stabilize_seconds=10.0, sample_steps=20, sample_interval=0.25,
                 cv_weight=10.0, food_weight=5.0, sat_penalty=1.0):
        self.base_url = base_url
        self.global_gain = global_gain
        self.tau_decay = tau_decay
        self.stabilize_seconds = stabilize_seconds
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval
        self.cv_weight = cv_weight
        self.food_weight = food_weight
        self.sat_penalty = sat_penalty
        self._last_food_distance = None
        self._episode_count = 0
    
    def _get_json(self, path):
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print("  [WARN] GET %s failed: %s" % (path, e))
            return None
    
    def _post_json(self, path, data):
        url = self.base_url + path
        try:
            payload = json.dumps(data).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={'Content-Type': 'application/json'},
                                         method='POST')
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print("  [WARN] POST %s failed: %s" % (path, e))
            return None
    
    def reset(self, seed=None):
        self._post_json('/api/params', {
            'global_gain': self.global_gain,
            'tau_decay': self.tau_decay,
            'per_joint_scales': {},
        })
        state = self._get_state()
        if state is None:
            state = np.zeros(self.STATE_DIM, dtype=np.float32)
        else:
            self._last_food_distance = state[self.N_JOINTS]
        self._episode_count += 1
        return state, {}
    
    def step(self, action):
        action_np = np.asarray(action, dtype=np.float64)
        # Sigmoid squash: action logits -> (0,1) -> [0.5, 2.0]
        multipliers = 0.5 + 1.5 / (1.0 + np.exp(-action_np))
        multipliers = np.clip(multipliers, 0.5, 2.0)
        
        per_joint_scales = {}
        for i, jname in enumerate(self.JOINT_NAMES):
            per_joint_scales[jname] = float(multipliers[i])
        
        result = self._post_json('/api/params', {'per_joint_scales': per_joint_scales})
        if result is None:
            return np.zeros(self.STATE_DIM, dtype=np.float32), -10.0, True, False, {'error': 'api_failure'}
        
        time.sleep(self.stabilize_seconds)
        
        torque_samples = []
        status_samples = []
        for _ in range(self.sample_steps):
            torque_data = self._get_json('/api/torque')
            status_data = self._get_json('/api/status')
            if torque_data and 'joints' in torque_data:
                torques = [torque_data['joints'].get(jname, 0.0) for jname in self.JOINT_NAMES]
                torque_samples.append(torques)
            if status_data:
                status_samples.append(status_data)
            time.sleep(self.sample_interval)
        
        if not torque_samples:
            return np.zeros(self.STATE_DIM, dtype=np.float32), -10.0, True, False, {'error': 'no_samples'}
        
        torque_cv = self._compute_torque_cv(torque_samples)
        food_dist = status_samples[-1].get('metrics', {}).get('food_distance', 0.0) if status_samples else 1.0
        
        cv_reward = torque_cv * self.cv_weight
        if self._last_food_distance is not None:
            food_delta = self._last_food_distance - food_dist
            food_reward = max(0, food_delta) * self.food_weight
        else:
            food_delta = 0.0
            food_reward = 0.0
        
        sat_count = sum(1 for sample in torque_samples
                       for t in sample if abs(abs(t) - 1.0) < 0.001)
        total_count = len(torque_samples) * self.N_JOINTS
        sat_ratio = sat_count / total_count if total_count > 0 else 0
        saturation_penalty = sat_ratio * self.sat_penalty * 5.0
        
        reward = cv_reward + food_reward - saturation_penalty
        
        state = self._build_state(torque_samples, status_samples)
        self._last_food_distance = food_dist
        
        info = {
            'torque_cv': float(torque_cv),
            'food_distance': float(food_dist),
            'food_delta': float(food_delta),
            'cv_reward': float(cv_reward),
            'food_reward': float(food_reward),
            'saturation_penalty': float(saturation_penalty),
            'sat_ratio': float(sat_ratio),
            'multipliers': multipliers.tolist(),
            'multiplier_mean': float(np.mean(multipliers)),
            'multiplier_std': float(np.std(multipliers)),
        }
        return state, reward, True, False, info
    
    def _compute_torque_cv(self, torque_samples):
        if len(torque_samples) < 2:
            return 0.0
        arr = np.array(torque_samples)
        abs_arr = np.abs(arr)
        joint_means = np.mean(abs_arr, axis=0)
        joint_stds = np.std(abs_arr, axis=0)
        valid = joint_means > 1e-6
        if not valid.any():
            return 0.0
        joint_cvs = np.zeros_like(joint_means)
        joint_cvs[valid] = joint_stds[valid] / joint_means[valid]
        return float(np.mean(joint_cvs))
    
    def _get_state(self):
        torque_data = self._get_json('/api/torque')
        status_data = self._get_json('/api/status')
        if torque_data and status_data:
            torques = [torque_data['joints'].get(jname, 0.0) for jname in self.JOINT_NAMES]
            return self._build_observation(torques, status_data)
        return None
    
    def _build_state(self, torque_samples, status_samples):
        if not torque_samples or not status_samples:
            return np.zeros(self.STATE_DIM, dtype=np.float32)
        arr = np.array(torque_samples)
        mean_torques = np.mean(arr, axis=0)
        last_status = status_samples[-1]
        return self._build_observation(mean_torques, last_status)
    
    def _build_observation(self, torques, status_data):
        metrics = status_data.get('metrics', {})
        obs = list(torques[:self.N_JOINTS])
        obs.append(metrics.get('food_distance', 1.0))
        obs.append(metrics.get('z_height', 0.15))
        obs.append(math.log1p(metrics.get('fired_neurons', 0)) / 10.0)
        obs.append(metrics.get('dn_matches', 0) / 50.0)
        obs.append(metrics.get('mns_activated', 0) / 200.0)
        obs.append(metrics.get('active_joints', 0) / 36.0)
        obs.append(metrics.get('wall_distance', 5.0) / 10.0)
        return np.array(obs, dtype=np.float32)
    
    def get_joint_names(self):
        return self.JOINT_NAMES


if __name__ == '__main__':
    print('Brain2Env smoke test')
    env = Brain2Env(stabilize_seconds=2.0, sample_steps=5, sample_interval=0.1)
    obs, _ = env.reset()
    print("  State dim: %d, Food dist: %.3f" % (len(obs), obs[36]))
    action = np.full(36, -0.693, dtype=np.float32)
    obs, reward, done, trunc, info = env.step(action)
    print("  Reward: %.4f, CV: %.4f" % (reward, info.get('torque_cv', 0)))
    print("  Multiplier mean: %.4f" % info.get('multiplier_mean', 0))
    print('Smoke test PASSED')
