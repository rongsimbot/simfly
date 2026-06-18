#!/usr/bin/env python3
import mujoco
"""
Phase 10: Mechanosensory Module — Ground Contact & Proprioception
Detects ground contact and body forces from MuJoCo simulation.
Maps to Drosophila mechanosensory bristles, campaniform sensilla,
and chordotonal organs.

Architecture:
  MuJoCo contact forces → touch bristle activation
  Joint angles/velocities → chordotonal organ activation
  → sensory injector → NIRON → DNs
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class GroundContactDetector:
    """Detect ground contact from MuJoCo contact data.
    
    Monitors leg contacts with the ground plane (z ≈ 0).
    """
    
    def __init__(self, model, data, contact_threshold: float = 0.001):
        self.model = model
        self.data = data
        self.contact_threshold = contact_threshold  # N
        
        # Track which bodies are leg parts
        self._leg_bodies = []
        self._init_leg_bodies()
    
    def _init_leg_bodies(self):
        """Identify leg-related bodies in the model."""
        leg_keywords = ['femur', 'tibia', 'tarsus', 'coxa', 'trochanter', 'leg', 'foot', 'claw']
        for i in range(self.model.nbody):
            name = self.model.body(i).name.lower() if hasattr(self.model.body(i), 'name') else ''
            if any(kw in name for kw in leg_keywords):
                self._leg_bodies.append(i)
    
    def detect_contacts(self) -> List[Dict]:
        """Detect ground contacts.
        
        Returns:
            List of contact dicts with body_id, force, position.
        """
        contacts = []
        ncon = self.data.ncon
        
        for i in range(ncon):
            contact = self.data.contact[i]
            
            # Get force magnitude
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_mag = np.linalg.norm(force[:3])
            
            if force_mag < self.contact_threshold:
                continue
            
            # Get contact position
            pos = contact.pos.copy()
            
            # Check if it's a leg contact (body1 or body2 is a leg body)
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            is_leg = body1 in self._leg_bodies or body2 in self._leg_bodies
            
            contacts.append({
                'body1': int(body1),
                'body2': int(body2),
                'geom1': int(geom1),
                'geom2': int(geom2),
                'force': float(force_mag),
                'position': pos.tolist(),
                'is_leg_contact': is_leg,
                'is_ground': pos[2] < 0.01,  # Ground is at z≈0
            })
        
        return contacts
    
    def get_leg_contact_summary(self) -> Dict:
        """Summarize leg ground contacts.
        
        Returns:
            {num_legs_on_ground, total_force, left_contacts, right_contacts}
        """
        contacts = self.detect_contacts()
        leg_contacts = [c for c in contacts if c['is_leg_contact']]
        ground_contacts = [c for c in leg_contacts if c['is_ground']]
        
        total_force = sum(c['force'] for c in ground_contacts)
        
        # Left vs right (based on position Y)
        left_contacts = sum(1 for c in ground_contacts if c['position'][1] < 0)
        right_contacts = sum(1 for c in ground_contacts if c['position'][1] >= 0)
        
        return {
            'num_leg_contacts': len(ground_contacts),
            'total_contact_force': total_force,
            'left_side_contacts': left_contacts,
            'right_side_contacts': right_contacts,
            'is_on_ground': len(ground_contacts) > 0,
        }


class ProprioceptionReader:
    """Read proprioceptive data from MuJoCo joint states.
    
    Encodes joint angles and velocities as chordotonal organ responses.
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.njnt = model.njnt
        
        # Identify leg joints
        self._leg_joints = []
        leg_keywords = ['femur', 'tibia', 'coxa', 'trochanter', 'tarsus', 'leg', 'foot']
        for i in range(self.njnt):
            name = self.model.joint(i).name.lower() if hasattr(self.model.joint(i), 'name') else ''
            if any(kw in name for kw in leg_keywords):
                self._leg_joints.append(i)
        
        # If no leg joints identified, use all joints
        if not self._leg_joints:
            self._leg_joints = list(range(min(self.njnt, 50)))
    
    def read(self) -> Dict:
        """Read joint angles and velocities.
        
        Returns:
            {joint_angles: [...], joint_velocities: [...], 
             leg_angles: [...], leg_velocities: [...], ...}
        """
        # Get all free joint DoFs for position/orientation
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Extract leg-specific data
        leg_angles = []
        leg_velocities = []
        
        # MuJoCo joint qpos addresses
        for jnt_id in self._leg_joints:
            addr = self.model.jnt_qposadr[jnt_id]
            if addr >= 0 and addr < len(qpos):
                leg_angles.append(float(qpos[addr]))
            addr_vel = self.model.jnt_dofadr[jnt_id]
            if addr_vel >= 0 and addr_vel < len(qvel):
                leg_velocities.append(float(qvel[addr_vel]))
        
        # Body position and velocity (from free joint)
        body_pos = qpos[0:3] if len(qpos) >= 3 else np.zeros(3)
        body_vel = qvel[0:3] if len(qvel) >= 3 else np.zeros(3)
        
        return {
            'body_position': body_pos.tolist(),
            'body_velocity': body_vel.tolist(),
            'body_speed': float(np.linalg.norm(body_vel)),
            'joint_angles': qpos.tolist() if len(qpos) < 100 else qpos[:100].tolist(),
            'joint_velocities': qvel.tolist() if len(qvel) < 100 else qvel[:100].tolist(),
            'leg_angles': leg_angles,
            'leg_velocities': leg_velocities,
            'num_leg_joints': len(self._leg_joints),
        }
    
    def get_chordotonal_input(self) -> Tuple[List[float], List[float]]:
        """Get chordotonal organ input values.
        
        Returns:
            (joint_angles, joint_velocities) for sensory injector.
        """
        data = self.read()
        return data['leg_angles'], data['leg_velocities']


class MechanoSensorySystem:
    """Combined mechanosensory system.
    
    Coordinates ground contact detection and proprioception.
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.contact_detector = GroundContactDetector(model, data)
        self.proprio_reader = ProprioceptionReader(model, data)
    
    def read(self) -> Dict:
        """Read all mechanosensory data.
        
        Returns:
            Combined mechanosensory dict.
        """
        contacts = self.contact_detector.get_leg_contact_summary()
        proprio = self.proprio_reader.read()
        
        return {
            **contacts,
            **proprio,
            'is_moving': proprio['body_speed'] > 0.001,
        }
    
    def get_touch_input(self) -> float:
        """Get touch/contact input level [0, 1] for sensory injector."""
        summary = self.contact_detector.get_leg_contact_summary()
        return min(1.0, summary['total_contact_force'] / 1.0)
    
    def get_proprioceptive_input(self) -> Tuple[List[float], List[float]]:
        """Get proprioceptive input for sensory injector."""
        return self.proprio_reader.get_chordotonal_input()
