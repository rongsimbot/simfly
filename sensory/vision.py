#!/usr/bin/env python3
"""
Phase 10: Sensory-Driven Vision Module — Obstacle Detection & Looming
Detects obstacles in the fly's visual field using MuJoCo ray-casting.
Maps to Drosophila LC4 (looming) and LC10 (small object) neuron responses.

FIXED (2026-06-11):
  1. Filter self-geoms: exclude fly body parts from ray detection
  2. Use sim_time for looming: replace perf_counter() with sim_time_ms
  3. Configurable wall/arena bounds: pass arena_bounds to ObstacleDetector
  4. Recalibrate contrast: varies linearly with distance; food marker detection

Architecture:
  MuJoCo ray-cast → obstacle detection → looming computation
  → photoreceptor activation → sensory injector → NIRON → DNs
"""

import math
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

# LC4 looming neuron parameters (von Reyn et al., 2014; Klapoetke et al., 2017)
LC4_LOOMING_THRESHOLD = 0.05   # Minimum angular expansion rate (rad/s) to trigger LC4
LC4_BASELINE = 0.0             # Baseline activity
LC4_GAIN = 50.0                # Hz per looming unit
LC4_MAX_RATE = 200.0           # Saturation

# LC10 small object detection (Keleş & Frye, 2017)
LC10_OBJECT_ANGLE_MIN = 1.0    # degrees — minimum object angular size
LC10_OBJECT_ANGLE_MAX = 30.0   # degrees
LC10_GAIN = 40.0               # Hz per object

# Photoreceptor parameters
PR_BASELINE = 10.0             # Hz baseline in darkness
PR_GAIN = 30.0                 # Hz per contrast unit
PR_ADAPTATION_TAU = 100.0      # ms


class ObstacleDetector:
    """Detect obstacles using MuJoCo ray-casting from fly head position.
    
    Casts rays in a forward cone (mimicking compound eye ommatidial field)
    and reports distance, angle, and size of detected obstacles.
    
    FIXES:
    - Filters self-geoms (fly body parts) from ray detection
    - Configurable wall position via arena_bounds
    - Uses sim time instead of perf_counter()
    """
    
    def __init__(
        self,
        model,
        data,
        num_rays: int = 10,
        fov_degrees: float = 180.0,
        max_distance: float = 10.0,
        ray_length: float = 10.0,
        arena_bounds: Optional[Dict[str, float]] = None,
        food_position: Optional[Tuple[float, float, float]] = None,
    ):
        self.model = model
        self.data = data
        self.num_rays = num_rays
        self.fov_rad = math.radians(fov_degrees)
        self.max_distance = max_distance
        
        # Arena bounds (configurable wall positions)
        self.arena_bounds = arena_bounds or {
            'x_min': -10.0, 'x_max': 5.0,
            'y_min': -5.0, 'y_max': 5.0,
            'z_min': -1.0, 'z_max': 5.0,
        }
        
        # Food position for visual food marker
        self.food_position = food_position
        
        # Pre-compute ray directions in fly body frame
        self.ray_directions = []
        half_fov = self.fov_rad / 2
        for i in range(num_rays):
            angle = -half_fov + i * self.fov_rad / max(1, num_rays - 1)
            dx = math.cos(angle)
            dy = math.sin(angle)
            dz = 0.0
            self.ray_directions.append(np.array([dx, dy, dz], dtype=np.float64))
        
        # History for looming detection (store (sim_time_ms, distance))
        self._distance_history: List[Tuple[float, float]] = []
        self._history_max = 20
        self._sim_time_ms: float = 0.0
    
    def _get_head_position(self) -> np.ndarray:
        """Get the fly's head position from MuJoCo data."""
        for name in ['head', 'Head', 'head_body']:
            try:
                body_id = self.model.body(name).id
                return self.data.body(body_id).xpos.copy()
            except KeyError:
                continue
        if self.data.geom_xpos.shape[0] > 0:
            return self.data.geom_xpos[0].copy()
        return np.array([0.0, 0.0, 0.0])
    
    def _get_fly_orientation(self) -> np.ndarray:
        """Get fly forward direction from body orientation."""
        try:
            body_id = self.model.body('head').id if 'head' in [self.model.body(i).name for i in range(self.model.nbody)] else 0
        except:
            body_id = 0
        mat = self.data.body(body_id).xmat.reshape(3, 3)
        forward = mat[:, 0]
        return forward
    
    def _get_fly_geom_ids(self) -> Set[int]:
        """FIX #1: Collect all geom IDs belonging to the fly body.
        
        These should be excluded from ray intersection tests to prevent
        the fly from detecting its own body parts as obstacles.
        """
        fly_geoms: Set[int] = set()
        try:
            # Collect all body IDs that belong to the fly
            fly_body_names = set()
            for i in range(self.model.nbody):
                name = self.model.body(i).name if hasattr(self.model.body(i), 'name') else ''
                if name and not name.startswith('world') and not name.startswith('floor') and not name.startswith('wall'):
                    fly_body_names.add(i)
            
            # Map bodies to their geoms
            for i in range(self.model.ngeom):
                body_id = self.model.geom_bodyid[i] if hasattr(self.model, 'geom_bodyid') else -1
                if body_id in fly_body_names:
                    fly_geoms.add(i)
        except Exception:
            pass
        return fly_geoms
    
    def _compute_wall_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> float:
        """FIX #3: Compute ray-arena-wall intersection using configurable bounds.
        
        Returns minimum positive distance to any arena wall, or max_distance.
        """
        min_dist = self.max_distance
        bounds = self.arena_bounds
        
        # Check intersection with each arena wall plane
        # x_min wall
        if ray_dir[0] < -0.001:
            d = (bounds['x_min'] - ray_origin[0]) / ray_dir[0]
            if 0 < d < min_dist:
                y_hit = ray_origin[1] + d * ray_dir[1]
                if bounds['y_min'] <= y_hit <= bounds['y_max']:
                    min_dist = d
        
        # x_max wall (forward wall)
        if ray_dir[0] > 0.001:
            d = (bounds['x_max'] - ray_origin[0]) / ray_dir[0]
            if 0 < d < min_dist:
                y_hit = ray_origin[1] + d * ray_dir[1]
                if bounds['y_min'] <= y_hit <= bounds['y_max']:
                    min_dist = d
        
        # y walls
        if ray_dir[1] > 0.001:
            d = (bounds['y_max'] - ray_origin[1]) / ray_dir[1]
            if 0 < d < min_dist:
                x_hit = ray_origin[0] + d * ray_dir[0]
                if bounds['x_min'] <= x_hit <= bounds['x_max']:
                    min_dist = d
        
        if ray_dir[1] < -0.001:
            d = (bounds['y_min'] - ray_origin[1]) / ray_dir[1]
            if 0 < d < min_dist:
                x_hit = ray_origin[0] + d * ray_dir[0]
                if bounds['x_min'] <= x_hit <= bounds['x_max']:
                    min_dist = d
        
        return min_dist
    
    def detect(self, sim_time_ms: float = 0.0) -> List[Dict]:
        """Cast rays and detect obstacles.
        
        Args:
            sim_time_ms: Simulation time in ms (FIX #2: use sim time, not wall clock)
        
        Returns:
            List of dicts: {distance, angle_deg, hit, geom_id, is_wall, is_food}
        """
        head_pos = self._get_head_position()
        fly_geoms = self._get_fly_geom_ids()  # FIX #1
        
        # Get body orientation for ray transformation
        try:
            body_id = self.model.body('head').id if 'head' in [self.model.body(i).name for i in range(self.model.nbody)] else 0
        except:
            body_id = 0
        xmat = self.data.body(body_id).xmat.reshape(3, 3)
        
        obstacles = []
        
        for i, ray_dir_body in enumerate(self.ray_directions):
            # Transform ray direction to world frame
            ray_dir_world = xmat @ ray_dir_body
            
            # Normalize
            norm = np.linalg.norm(ray_dir_world)
            if norm > 0:
                ray_dir_world = ray_dir_world / norm
            
            # Find nearest geom intersection (FIX #1: exclude fly body geoms)
            min_dist = self.max_distance
            hit_geom = -1
            
            for g in range(self.data.geom_xpos.shape[0]):
                if g in fly_geoms:  # FIX #1: skip fly body parts
                    continue
                
                geom_pos = self.data.geom_xpos[g]
                to_geom = geom_pos - head_pos
                dist = np.linalg.norm(to_geom)
                
                if dist < 0.001 or dist > self.max_distance:
                    continue
                
                to_geom_dir = to_geom / dist
                dot = np.dot(ray_dir_world, to_geom_dir)
                
                if dot > 0.996:  # cos(5°)
                    if dist < min_dist:
                        min_dist = dist
                        hit_geom = g
            
            # FIX #3: Configurable wall detection
            if self.arena_bounds:
                wall_dist = self._compute_wall_intersection(head_pos, ray_dir_world)
                if wall_dist < min_dist:
                    min_dist = wall_dist
                    hit_geom = -2  # -2 = arena wall
            
            angle_deg = math.degrees(-self.fov_rad/2 + i * self.fov_rad / max(1, self.num_rays - 1))
            
            # FIX: Check if this ray points toward food (visual food marker)
            is_food = False
            if self.food_position is not None:
                food_vec = np.array(self.food_position[:2]) - head_pos[:2]
                food_dist = np.linalg.norm(food_vec)
                if food_dist > 0.001:
                    food_dir = np.array([food_vec[0], food_vec[1], 0.0])
                    food_dir = food_dir / np.linalg.norm(food_dir)
                    food_dot = np.dot(ray_dir_world[:2], food_dir[:2])
                    if food_dot > 0.95:  # Within ~18° of food
                        is_food = True
            
            obstacles.append({
                'distance': min_dist,
                'angle_deg': angle_deg,
                'hit': min_dist < self.max_distance,
                'geom_id': hit_geom if hit_geom >= 0 else None,
                'is_wall': hit_geom == -2,
                'is_food': is_food,  # Visual food marker
            })
        
        # FIX #2: Use sim time for history (not perf_counter)
        self._sim_time_ms = sim_time_ms
        min_forward_dist = min(o['distance'] for o in obstacles if o.get('is_wall') or o['hit'])
        self._distance_history.append((sim_time_ms, min_forward_dist))
        if len(self._distance_history) > self._history_max:
            self._distance_history.pop(0)
        
        return obstacles
    
    def get_nearest_distance(self) -> float:
        """Get distance to nearest obstacle in front."""
        obstacles = self.detect(sim_time_ms=self._sim_time_ms)
        forward_obs = [o for o in obstacles if abs(o['angle_deg']) < 30]
        if forward_obs:
            return min(o['distance'] for o in forward_obs)
        return self.max_distance


class LoomingDetector:
    """Detect looming stimuli from obstacle approach.
    
    Looming = angular expansion rate dθ/dt ≈ (object_size / distance²) × approach_velocity
    Maps to Drosophila LC4 neuron response.
    
    FIX #2: Uses sim time (dt_ms) instead of perf_counter() for deterministic looming.
    """
    
    def __init__(self, history_size: int = 20):
        self.history: List[Tuple[float, float]] = []  # (sim_time_ms, distance)
        self.history_size = history_size
        self._looming_intensity = 0.0
        self._lc4_rate = 0.0
    
    def update(self, distance: float, sim_time_ms: float = 0.0, object_size: float = 0.01) -> float:
        """Update looming detection using sim time.
        
        Args:
            distance: Distance to nearest obstacle (m).
            sim_time_ms: Simulation time in ms (FIX #2).
            object_size: Estimated object size (m).
            
        Returns:
            Looming intensity [0, 1].
        """
        self.history.append((sim_time_ms, distance))
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        looming = 0.0
        
        if len(self.history) >= 2 and distance < 999:
            t0, d0 = self.history[0]
            t1, d1 = self.history[-1]
            dt = (t1 - t0) / 1000.0  # Convert ms to seconds
            if dt > 0.0001:
                approach_velocity = (d0 - d1) / dt  # Positive = approaching
                
                if approach_velocity > 0 and distance > 0.001:
                    # Angular expansion rate: dθ/dt ≈ size × v / d²
                    angular_expansion = (object_size * approach_velocity) / (distance * distance)
                    looming = min(1.0, angular_expansion / LC4_LOOMING_THRESHOLD)
        
        self._looming_intensity = looming
        
        # LC4 neuron response: rectified, saturating
        if looming > 0:
            lv = looming / (1.0 + math.exp(-10 * (looming - 0.3)))
            self._lc4_rate = min(LC4_MAX_RATE, LC4_GAIN * lv)
        else:
            self._lc4_rate = max(0, self._lc4_rate * 0.9)
        
        return looming
    
    @property
    def lc4_firing_rate(self) -> float:
        return self._lc4_rate
    
    @property
    def intensity(self) -> float:
        return self._looming_intensity


class SmallObjectDetector:
    """Detect small moving objects in visual field.
    
    Maps to LC10/LPLC2 neuron responses (Keleş & Frye, 2017).
    """
    
    def __init__(self, num_regions: int = 5):
        self.num_regions = num_regions
        self._object_presence = np.zeros(num_regions)
        self._lc10_rates = np.zeros(num_regions)
    
    def update(self, obstacles: List[Dict]) -> np.ndarray:
        """Update small object detection from obstacle data."""
        self._object_presence = np.zeros(self.num_regions)
        
        for obs in obstacles:
            if not obs['hit'] or obs['distance'] <= 0.001:
                continue
            
            angular_size_rad = 2 * math.atan(0.005 / obs['distance'])
            angular_size_deg = math.degrees(angular_size_rad)
            
            if LC10_OBJECT_ANGLE_MIN <= angular_size_deg <= LC10_OBJECT_ANGLE_MAX:
                region = int((obs['angle_deg'] + 45) / 90 * self.num_regions)
                region = max(0, min(self.num_regions - 1, region))
                
                response = min(1.0, angular_size_deg / LC10_OBJECT_ANGLE_MAX)
                self._object_presence[region] = max(self._object_presence[region], response)
        
        self._lc10_rates = self._object_presence * LC10_GAIN
        return self._object_presence
    
    @property
    def lc10_firing_rates(self) -> np.ndarray:
        return self._lc10_rates


class FlyVision:
    """Main vision module combining all detectors.
    
    Provides photoreceptor input and visual neuron responses 
    for the sensory injector.
    
    FIX #4: Recalibrated contrast — varies linearly with distance to obstacles,
            walls, and food. Food marker added as visual attractant.
    """
    
    def __init__(self, model, data, num_rays: int = 10,
                 arena_bounds: Optional[Dict[str, float]] = None,
                 food_position: Optional[Tuple[float, float, float]] = None):
        self.model = model
        self.data = data
        self.obstacle_detector = ObstacleDetector(
            model, data, num_rays=num_rays,
            arena_bounds=arena_bounds,
            food_position=food_position,
        )
        self.looming_detector = LoomingDetector()
        self.small_object_detector = SmallObjectDetector()
        
        # Food position for visual attraction
        self.food_position = food_position
        
        # Adaptation state
        self._left_adaptation = 0.5
        self._right_adaptation = 0.5
        self._adaptation_tau = 100.0  # ms
    
    def read(self, dt_ms: float = 1.0, sim_time_ms: float = 0.0) -> Dict:
        """Read visual scene and compute all responses.
        
        Args:
            dt_ms: Time step in ms.
            sim_time_ms: Simulation time in ms (for looming).
        
        Returns:
            Dict with all visual sensory data.
        """
        obstacles = self.obstacle_detector.detect(sim_time_ms=sim_time_ms)
        
        # Split obstacles by left/right visual field
        left_obs = [o for o in obstacles if o['angle_deg'] < -10]
        right_obs = [o for o in obstacles if o['angle_deg'] > 10]
        center_obs = [o for o in obstacles if abs(o['angle_deg']) <= 10]
        
        # Nearest distances per field
        left_dist = min((o['distance'] for o in left_obs if o['hit']), default=10.0)
        right_dist = min((o['distance'] for o in right_obs if o['hit']), default=10.0)
        center_dist = min((o['distance'] for o in center_obs if o['hit']), default=10.0)
        nearest_dist = min(left_dist, right_dist, center_dist)
        
        # FIX #4: Recalibrated contrast — LINEAR with distance
        # contrast = 1.0 when obstacle is at 0m, 0.0 at 10m+
        # Food rays get a bonus (visual food marker)
        has_wall = any(o.get('is_wall', False) for o in obstacles)
        has_food_ray = any(o.get('is_food', False) for o in obstacles)
        wall_dist = min((o['distance'] for o in obstacles if o.get('is_wall', False)), default=10.0)
        
        # FIX #4: Calibrate contrast — linear decay from distance
        # At distance=0, contrast=1.0; at distance=5m, contrast=0.0
        MAX_CONTRAST_DISTANCE = 5.0  # meters
        FOOD_CONTRAST_BOOST = 1.5   # food marker is brighter
        
        if wall_dist < MAX_CONTRAST_DISTANCE:
            contrast = 1.0 - min(1.0, wall_dist / MAX_CONTRAST_DISTANCE)
        else:
            contrast = 0.0
        
        # Food visual marker: rays hitting food get boosted brightness
        food_brightness = 0.0
        if has_food_ray and self.food_position is not None:
            # Calculate distance to food for brightness
            head_pos = self.obstacle_detector._get_head_position()
            food_vec = np.array([
                self.food_position[0] - head_pos[0],
                self.food_position[1] - head_pos[1],
                self.food_position[2] - head_pos[2],
            ])
            food_dist = np.linalg.norm(food_vec)
            food_brightness = 1.0 - min(1.0, food_dist / MAX_CONTRAST_DISTANCE)
            food_brightness *= FOOD_CONTRAST_BOOST
        
        # Apply contrast to each visual field
        # FIX #4: Make contrast vary with distance per visual field
        def field_contrast(obs_list, default_dist=10.0):
            nearest = min((o['distance'] for o in obs_list if o['hit']), default=default_dist)
            has_food = any(o.get('is_food', False) for o in obs_list)
            if nearest < MAX_CONTRAST_DISTANCE:
                c = 1.0 - min(1.0, nearest / MAX_CONTRAST_DISTANCE)
            else:
                c = 0.0
            if has_food:
                c = max(c, food_brightness)  # Food marker boosts contrast
            return c
        
        left_brightness = field_contrast(left_obs)
        right_brightness = field_contrast(right_obs)
        center_brightness = field_contrast(center_obs)
        
        # Update adaptation
        decay = math.exp(-dt_ms / self._adaptation_tau)
        self._left_adaptation = self._left_adaptation * decay + left_brightness * (1 - decay)
        self._right_adaptation = self._right_adaptation * decay + right_brightness * (1 - decay)
        
        # FIX #2: Looming detection with sim time
        looming_intensity = self.looming_detector.update(
            nearest_dist, sim_time_ms=sim_time_ms, object_size=0.5
        )
        
        # Small object detection
        object_presence = self.small_object_detector.update(obstacles)
        
        return {
            'obstacles': obstacles,
            'nearest_distance': nearest_dist,
            'wall_distance': wall_dist,
            'has_wall': has_wall,
            'has_food_visual': has_food_ray,
            'food_brightness': food_brightness,
            'contrast': contrast,
            'left_eye_brightness': left_brightness,
            'right_eye_brightness': right_brightness,
            'center_brightness': center_brightness,
            'left_adaptation': self._left_adaptation,
            'right_adaptation': self._right_adaptation,
            'looming_intensity': looming_intensity,
            'lc4_rate': self.looming_detector.lc4_firing_rate,
            'small_object_presence': object_presence.tolist(),
            'lc10_rates': self.small_object_detector.lc10_firing_rates.tolist(),
        }
    
    def get_photoreceptor_input(self) -> Dict[str, float]:
        """Get photoreceptor activation values for sensory injector."""
        vision_data = self.read()
        return {
            'left_eye_brightness': vision_data['left_eye_brightness'],
            'right_eye_brightness': vision_data['right_eye_brightness'],
            'scene_brightness': max(vision_data['left_eye_brightness'], vision_data['right_eye_brightness']),
            'contrast': vision_data['contrast'],
            'lc4_looming_rate': vision_data['lc4_rate'],
            'food_visual': vision_data.get('food_brightness', 0.0),
        }
    
    def get_fly_view_image(self, width: int = 320, height: int = 240) -> 'Image':
        """Generate a visualization of what the fly sees.
        
        Creates a panoramic image showing obstacle distances, brightness,
        food markers, and looming from the fly's perspective.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None
        
        obstacles = self.obstacle_detector.detect()
        
        img = Image.new('RGB', (width, height), color=(5, 5, 20))
        draw = ImageDraw.Draw(img)
        
        if not obstacles:
            draw.text((10, height//2), "No visual data", fill=(100,100,100))
            return img
        
        # Draw panoramic view
        bar_width = width / max(1, len(obstacles))
        
        for i, obs in enumerate(obstacles):
            x = int(i * bar_width)
            dist = obs['distance']
            
            # Color = brightness (closer = brighter)
            if dist < self.obstacle_detector.max_distance:
                brightness = int(255 * (1.0 - min(1.0, dist / 5.0)))
                if obs.get('is_food'):
                    color = (brightness, brightness//3, brightness//3)  # Reddish for food
                elif obs.get('is_wall'):
                    color = (brightness//2, brightness//2, brightness)  # Blueish for walls
                else:
                    color = (brightness//2, brightness, brightness//2)  # Greenish for objects
            else:
                color = (10, 10, 15)  # Dark = nothing
            
            bar_h = int(height * (1.0 - min(1.0, dist / 10.0)))
            y0 = height - bar_h
            draw.rectangle([x, y0, x + int(bar_width) + 1, height], fill=color)
            
            # Mark food rays
            if obs.get('is_food'):
                draw.ellipse([x+1, 5, x+int(bar_width)-1, 25], outline=(255, 100, 50), width=1)
        
        # Draw horizontal line for reference
        draw.line([0, height//2, width, height//2], fill=(40, 40, 60), width=1)
        draw.text((5, height-15), "DARK = void | GREEN = objects | BLUE = walls | RED = food", 
                  fill=(80, 80, 80))
        
        return img
