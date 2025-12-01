#!/usr/bin/env python3
"""
ML Control System for Resonant Plasma Fusion
Full simulation-ready implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SENSOR SIMULATION & FEATURE EXTRACTION
# ============================================================================

class PlasmaSensorSimulator:
    """Simulates 12 sensor inputs from plasma diagnostics"""
    
    def __init__(self, sampling_rate=1e6, seed=42):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        np.random.seed(seed)
        
        # Simulated sensor characteristics
        self.sensors = {
            'B_field': {'freq': 500e3, 'noise': 0.1, 'phase': 0.0},
            'E_field': {'freq': 500e3, 'noise': 0.15, 'phase': np.pi/2},
            'plasma_density': {'freq': 100e3, 'noise': 0.05, 'trend': 1e19},
            'T_electron': {'freq': 10e3, 'noise': 0.2, 'trend': 10.0},
            'T_ion': {'freq': 10e3, 'noise': 0.25, 'trend': 10.0},
            'dust_position_x': {'freq': 1e3, 'noise': 0.01, 'drift': 0.001},
            'dust_position_y': {'freq': 1e3, 'noise': 0.01, 'drift': 0.001},
            'dust_position_z': {'freq': 1e3, 'noise': 0.01, 'drift': 0.001},
            'acoustic': {'freq': 1e6, 'noise': 0.3, 'burst_prob': 0.01},
            'RF_noise': {'freq': 13.56e6, 'noise': 0.4, 'modulation': 7.83},
            'optical_emission': {'freq': 1e3, 'noise': 0.15, 'hotspot_prob': 0.005},
            'vacuum_pressure': {'freq': 100, 'noise': 0.02, 'drift': -1e-7}
        }
        
        self.time = 0.0
        self.hotspot_active = False
        self.hotspot_duration = 0
        
    def read_sensors(self, control_signals: Dict) -> Dict:
        """Generate simulated sensor readings based on control inputs"""
        self.time += self.dt
        
        readings = {}
        
        for sensor_name, params in self.sensors.items():
            # Base signal
            t = self.time
            
            if sensor_name == 'B_field':
                # B-field responds to RF power
                base = np.sin(2 * np.pi * params['freq'] * t + params['phase'])
                base *= (1.0 + 0.5 * control_signals.get('rf_amplitude', 0.5))
                
            elif sensor_name == 'E_field':
                # E-field with phase shift
                base = np.cos(2 * np.pi * params['freq'] * t + params['phase'])
                base *= (1.0 + 0.3 * control_signals.get('rf_amplitude', 0.5))
                
            elif sensor_name == 'plasma_density':
                # Density increases with RF power
                base = params['trend'] * (1.0 + 0.8 * control_signals.get('rf_amplitude', 0.5))
                
            elif sensor_name == 'T_electron':
                # Temperature responds to pulses
                pulse_effect = 1.0 if control_signals.get('pulse_active', False) else 0.3
                base = params['trend'] * (1.0 + 2.0 * control_signals.get('rf_amplitude', 0.5) * pulse_effect)
                
            elif 'dust_position' in sensor_name:
                # Dust moves with clustering effects
                drift = params['drift'] * t
                cluster = 0.1 * np.sin(2 * np.pi * 100 * t)  # Clustering oscillation
                base = drift + cluster
                
            elif sensor_name == 'acoustic':
                # Bursty acoustic noise
                base = np.random.randn()
                if np.random.random() < params['burst_prob']:
                    base *= 10.0  # Acoustic burst
                    
            elif sensor_name == 'RF_noise':
                # RF noise with Schumann modulation
                base = np.sin(2 * np.pi * params['freq'] * t)
                modulation = 0.1 * np.sin(2 * np.pi * params['modulation'] * t)
                base *= (1.0 + modulation)
                
            elif sensor_name == 'optical_emission':
                # Optical emission with occasional hot spots
                base = 1.0 + 0.5 * control_signals.get('rf_amplitude', 0.5)
                
                # Simulate hot spot event
                if self.hotspot_active:
                    base *= 5.0  # 5x emission during hot spot
                    self.hotspot_duration -= 1
                    if self.hotspot_duration <= 0:
                        self.hotspot_active = False
                elif np.random.random() < params['hotspot_prob']:
                    self.hotspot_active = True
                    self.hotspot_duration = np.random.randint(10, 100)  # μs duration
                    
            elif sensor_name == 'vacuum_pressure':
                # Vacuum improves with time
                base = 1e-6 + params['drift'] * t
                base = max(base, 1e-9)  # Don't go below ultimate vacuum
                
            else:
                base = 1.0
            
            # Add noise
            noise = params['noise'] * np.random.randn()
            readings[sensor_name] = base + noise
            
        return readings

class FeatureExtractor:
    """Extracts 512 features from raw sensor data"""
    
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
        self.sensor_buffer = deque(maxlen=1000)  # Store last 1000 readings
        
        # FFT parameters
        self.fft_window = 256
        self.fft_overlap = 128
        
    def extract_features(self, sensor_readings: Dict, history: List[Dict]) -> np.ndarray:
        """Extract 512-dimensional feature vector"""
        features = []
        
        # 1. Time-domain statistics for each sensor
        for sensor_name, value in sensor_readings.items():
            features.append(value)  # Current value
            features.append(np.mean([h[sensor_name] for h in history[-10:]]))  # 10-point MA
            features.append(np.std([h[sensor_name] for h in history[-50:]]))  # 50-point std
            
        # 2. Frequency domain features (simplified FFT)
        if len(history) >= self.fft_window:
            for sensor_name in ['B_field', 'E_field', 'RF_noise', 'acoustic']:
                values = [h[sensor_name] for h in list(history)[-self.fft_window:]]
                fft = np.abs(np.fft.rfft(values))
                features.append(np.mean(fft[:10]))  # Low freq mean
                features.append(np.max(fft))  # Peak amplitude
                features.append(np.argmax(fft))  # Peak frequency bin
                
        # 3. Cross-sensor correlations
        sensors_list = list(sensor_readings.keys())
        for i in range(min(5, len(sensors_list))):
            for j in range(i+1, min(6, len(sensors_list))):
                s1 = sensors_list[i]
                s2 = sensors_list[j]
                vals1 = [h[s1] for h in history[-100:]]
                vals2 = [h[s2] for h in history[-100:]]
                if len(vals1) > 10:
                    corr = np.corrcoef(vals1, vals2)[0,1]
                    features.append(corr if not np.isnan(corr) else 0.0)
                    
        # 4. Phase synchronization (B vs E fields)
        if 'B_field' in sensor_readings and 'E_field' in sensor_readings:
            b_vals = [h['B_field'] for h in history[-100:]]
            e_vals = [h['E_field'] for h in history[-100:]]
            if len(b_vals) > 10:
                # Simple phase difference metric
                phase_diff = np.mean(np.abs(np.array(b_vals) - np.array(e_vals)))
                features.append(phase_diff)
                
        # 5. Dust clustering metrics
        dust_positions = []
        for h in history[-100:]:
            if all(k in h for k in ['dust_position_x', 'dust_position_y', 'dust_position_z']):
                dust_positions.append([
                    h['dust_position_x'],
                    h['dust_position_y'], 
                    h['dust_position_z']
                ])
        
        if len(dust_positions) > 20:
            dust_array = np.array(dust_positions)
            # Cluster density (simplified)
            bbox_volume = np.prod(dust_array.ptp(axis=0) + 1e-10)
            if bbox_volume > 0:
                density = len(dust_array) / bbox_volume
                features.append(density)
                
                # Spatial spread
                features.append(np.std(dust_array[:,0]))  # X spread
                features.append(np.std(dust_array[:,1]))  # Y spread
                features.append(np.std(dust_array[:,2]))  # Z spread
                
        # 6. Hot spot indicators
        if 'optical_emission' in sensor_readings:
            recent_emission = [h['optical_emission'] for h in history[-20:]]
            features.append(sensor_readings['optical_emission'] / (np.mean(recent_emission) + 1e-10))
            features.append(np.max(recent_emission) - np.min(recent_emission))
            
        # 7. System health metrics
        features.append(sensor_readings.get('vacuum_pressure', 1e-6))
        features.append(sensor_readings.get('T_electron', 10.0) / 100.0)  # Normalized
        
        # Pad or truncate to desired dimension
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
            
        return np.array(features, dtype=np.float32)

# ============================================================================
# 2. NEURAL CONTROL NETWORK (LSTM + Attention)
# ============================================================================

class SafetyLayer(nn.Module):
    """Hard safety constraints for control outputs"""
    
    def __init__(self, max_rf_power=2.0, max_pulse_width=100.0, 
                 max_freq_deviation=0.1, max_phase_step=10.0):
        super().__init__()
        self.max_rf_power = max_rf_power  # kW
        self.max_pulse_width = max_pulse_width  # μs
        self.max_freq_deviation = max_freq_deviation  # Hz from 7.83
        self.max_phase_step = max_phase_step  # degrees per step
        
        self.last_controls = None
        
    def forward(self, raw_controls: torch.Tensor, state: Dict) -> torch.Tensor:
        """Apply safety constraints to raw control outputs"""
        if self.last_controls is None:
            self.last_controls = torch.zeros_like(raw_controls)
            
        safe_controls = raw_controls.clone()
        
        # 1. RF Power limits (0 to max_rf_power kW)
        safe_controls[0] = torch.clamp(safe_controls[0], 0.0, self.max_rf_power)
        
        # 2. Pulse width limits (10 to max_pulse_width μs)
        safe_controls[1] = torch.clamp(safe_controls[1], 10.0, self.max_pulse_width)
        
        # 3. Frequency modulation limits (7.83 ± max_freq_deviation Hz)
        base_freq = 7.83
        freq_control = safe_controls[2]
        safe_controls[2] = torch.clamp(freq_control, 
                                       base_freq - self.max_freq_deviation,
                                       base_freq + self.max_freq_deviation)
        
        # 4. Phase shift limits (0 to 360 degrees)
        safe_controls[3] = safe_controls[3] % 360.0
        
        # 5. Rate limiting (prevent abrupt changes)
        max_change = torch.tensor([0.5, 20.0, 0.05, 45.0])  # Max change per step
        changes = safe_controls - self.last_controls
        limited_changes = torch.clamp(changes, -max_change, max_change)
        safe_controls = self.last_controls + limited_changes
        
        # 6. Temperature-based constraints
        if 'T_dust' in state and state['T_dust'] > 2500:
            # Reduce RF power if dust too hot
            safe_controls[0] *= 0.5
            
        if 'Gamma' in state and state['Gamma'] > 250:
            # Reduce power if plasma unstable
            safe_controls[0] *= 0.3
            safe_controls[1] *= 0.8  # Shorter pulses
            
        self.last_controls = safe_controls.clone()
        return safe_controls

class PlasmaControlNet(nn.Module):
    """LSTM + Attention neural controller for plasma system"""
    
    def __init__(self, input_dim=512, hidden_dim=256, control_dim=4, 
                 num_layers=2, num_heads=8):
        super().__init__()
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Control policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.ln2,
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, control_dim)
        )
        
        # Safety layer
        self.safety_layer = SafetyLayer()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
                
    def forward(self, x: torch.Tensor, hidden_state: Tuple = None, 
                state_info: Dict = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            hidden_state: (h, c) for LSTM
            state_info: Dictionary with system state for safety layer
            
        Returns:
            controls: [batch_size, control_dim]
            attention_weights: [batch_size, seq_len, seq_len]
            new_hidden: Updated LSTM state
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        # lstm_out: [batch_size, seq_len, hidden_dim*2]
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_out: [batch_size, seq_len, hidden_dim*2]
        
        # Residual connection + layer norm
        attn_out = self.ln1(lstm_out + attn_out)
        
        # Use last timestep for control decision
        context = attn_out[:, -1, :]  # [batch_size, hidden_dim*2]
        
        # Generate raw control signals
        raw_controls = self.policy_net(context)  # [batch_size, control_dim]
        
        # Apply safety constraints if state info provided
        if state_info is not None:
            controls = torch.zeros_like(raw_controls)
            for i in range(batch_size):
                controls[i] = self.safety_layer(raw_controls[i].unsqueeze(0), 
                                                state_info)[0]
        else:
            controls = raw_controls
            
        return controls, attn_weights, new_hidden

# ============================================================================
# 3. REINFORCEMENT LEARNING ENVIRONMENT
# ============================================================================

class PlasmaRLEnvironment:
    """Reinforcement learning environment for plasma control"""
    
    def __init__(self):
        self.sensor_sim = PlasmaSensorSimulator(sampling_rate=1e6)
        self.feature_extractor = FeatureExtractor(feature_dim=512)
        self.sensor_history = []
        
        # State variables
        self.dust_mass = 1.0  # Normalized initial mass
        self.cluster_stability = 0.5
        self.hotspot_count = 0
        self.energy_consumption = 0.0
        self.step_count = 0
        
        # Target parameters
        self.target_gamma = 180.0
        self.max_dust_temp = 2500.0
        
        # Control limits
        self.control_bounds = {
            'rf_power': (0.0, 2.0),      # kW
            'pulse_width': (10.0, 100.0), # μs
            'mod_freq': (7.73, 7.93),    # Hz (7.83 ± 0.1)
            'phase': (0.0, 360.0)        # degrees
        }
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.sensor_sim = PlasmaSensorSimulator(sampling_rate=1e6)
        self.sensor_history = []
        self.dust_mass = 1.0
        self.cluster_stability = 0.5
        self.hotspot_count = 0
        self.energy_consumption = 0.0
        self.step_count = 0
        
        # Get initial observation
        for _ in range(100):  # Fill history buffer
            controls = {'rf_amplitude': 0.5, 'pulse_active': False}
            readings = self.sensor_sim.read_sensors(controls)
            self.sensor_history.append(readings)
            
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (feature vector)"""
        if len(self.sensor_history) == 0:
            return np.zeros(512, dtype=np.float32)
            
        current_readings = self.sensor_history[-1]
        features = self.feature_extractor.extract_features(
            current_readings, 
            self.sensor_history
        )
        return features
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one control step
        
        Args:
            action: [rf_power, pulse_width, mod_freq, phase]
            
        Returns:
            observation: Next state feature vector
            reward: Calculated reward
            done: Whether episode is done
            info: Additional information
        """
        self.step_count += 1
        
        # Decode action
        rf_power = np.clip(action[0], *self.control_bounds['rf_power'])
        pulse_width = np.clip(action[1], *self.control_bounds['pulse_width'])
        mod_freq = np.clip(action[2], *self.control_bounds['mod_freq'])
        phase = np.clip(action[3], *self.control_bounds['phase'])
        
        # Create control signals for sensor simulation
        control_signals = {
            'rf_amplitude': rf_power / 2.0,  # Normalize to [0, 1]
            'pulse_active': pulse_width > 20.0,  # Simple pulse trigger
            'pulse_width': pulse_width,
            'mod_freq': mod_freq,
            'phase': phase
        }
        
        # Get sensor readings
        readings = self.sensor_sim.read_sensors(control_signals)
        self.sensor_history.append(readings)
        
        # Update internal state (simplified physics)
        self._update_physics(action, readings)
        
        # Calculate reward
        reward = self._calculate_reward(readings)
        
        # Check termination conditions
        done = self._check_termination()
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Info dictionary
        info = {
            'dust_mass': self.dust_mass,
            'cluster_stability': self.cluster_stability,
            'hotspot_count': self.hotspot_count,
            'energy_consumption': self.energy_consumption,
            'rf_power': rf_power,
            'pulse_width': pulse_width
        }
        
        return next_obs, reward, done, info
    
    def _update_physics(self, action: np.ndarray, readings: Dict):
        """Update internal physics model (simplified)"""
        rf_power = action[0]
        
        # Dust mass loss (simplified model)
        T_e = readings.get('T_electron', 10.0)
        dust_temp = 300.0 + 0.1 * rf_power * T_e  # Simplified heating
        
        if dust_temp > self.max_dust_temp:
            mass_loss = 0.01 * (dust_temp - self.max_dust_temp) / 100.0
            self.dust_mass = max(0.0, self.dust_mass - mass_loss)
            
        # Cluster stability
        b_field_var = np.std([h.get('B_field', 0) for h in self.sensor_history[-20:]])
        self.cluster_stability = 0.9 * self.cluster_stability + 0.1 * (1.0 / (1.0 + b_field_var))
        self.cluster_stability = np.clip(self.cluster_stability, 0.0, 1.0)
        
        # Hot spot detection
        emission_ratio = readings.get('optical_emission', 1.0) / \
                        (np.mean([h.get('optical_emission', 1.0) for h in self.sensor_history[-20:]]) + 1e-10)
        if emission_ratio > 3.0:
            self.hotspot_count += 1
            
        # Energy consumption
        self.energy_consumption += rf_power * 0.01  # kW * timestep
        
    def _calculate_reward(self, readings: Dict) -> float:
        """Calculate reward based on current state"""
        reward = 0.0
        
        # 1. Dust preservation reward
        reward += 0.5 * self.dust_mass
        
        # 2. Cluster stability reward
        reward += 0.3 * self.cluster_stability
        
        # 3. Hot spot reward (but not too many)
        hotspot_reward = min(self.hotspot_count, 10) * 0.1
        reward += hotspot_reward
        
        # 4. Resonance quality (B and E field correlation)
        if 'B_field' in readings and 'E_field' in readings:
            recent_b = [h.get('B_field', 0) for h in self.sensor_history[-20:]]
            recent_e = [h.get('E_field', 0) for h in self.sensor_history[-20:]]
            if len(recent_b) > 5:
                correlation = np.corrcoef(recent_b, recent_e)[0,1]
                if not np.isnan(correlation):
                    reward += 0.2 * abs(correlation)
        
        # 5. Energy efficiency penalty
        reward -= 0.01 * self.energy_consumption
        
        # 6. Penalize extreme conditions
        T_e = readings.get('T_electron', 10.0)
        if T_e > 100.0:
            reward -= 0.1 * (T_e - 100.0) / 100.0
            
        if self.dust_mass < 0.5:
            reward -= 1.0
            
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if dust mostly gone
        if self.dust_mass < 0.1:
            return True
            
        # Terminate after max steps
        if self.step_count >= 1000:
            return True
            
        # Terminate if plasma unstable
        recent_gamma = [h.get('T_electron', 10.0) / h.get('plasma_density', 1e19) 
                       for h in self.sensor_history[-10:]]
        if len(recent_gamma) > 5 and np.mean(recent_gamma) > 1e-3:
            return True
            
        return False

# ============================================================================
# 4. PPO AGENT IMPLEMENTATION
# ============================================================================

class PPOBuffer:
    """Buffer for storing trajectory data for PPO"""
    
    def __init__(self, obs_dim, act_dim, size=1000, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, val, logp):
        """Store one timestep of data"""
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_val=0):
        """Finish trajectory and compute advantages"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Rewards-to-go
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sums"""
        return np.array([np.sum(x[i:] * (discount ** np.arange(len(x) - i))) 
                        for i in range(len(x))])
        
    def get(self):
        """Get all data from buffer"""
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, obs_dim=512, act_dim=4, hidden_dim=256, 
                 lr=3e-4, clip_ratio=0.2, train_iters=80, target_kl=0.01):
        
        # Policy network (actor)
        self.policy_net = PlasmaControlNet(input_dim=obs_dim, control_dim=act_dim)
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.act_dim = act_dim
        
        # Action distribution
        self.action_std = nn.Parameter(torch.ones(act_dim) * 0.5)
        
    def get_action(self, obs: np.ndarray, hidden_state=None) -> Tuple[np.ndarray, np.ndarray, float, Tuple]:
        """Get action from policy network"""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # obs_t: [1, 1, obs_dim]
        
        with torch.no_grad():
            # Get deterministic action mean
            action_mean, attn_weights, new_hidden = self.policy_net(obs_t, hidden_state)
            action_mean = action_mean.squeeze(0)  # [act_dim]
            
            # Add exploration noise
            action_std = torch.clamp(self.action_std, 0.1, 1.0)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            logp = dist.log_prob(action).sum()
            
            # Get value estimate
            value = self.value_net(obs_t.squeeze(1)).squeeze()
            
        return (action.numpy(), value.numpy(), logp.numpy(), 
                (new_hidden[0].numpy(), new_hidden[1].numpy()))
    
    def update(self, buffer: PPOBuffer):
        """Update policy and value networks using PPO"""
        data = buffer.get()
        
        # Convert to tensors
        obs = torch.as_tensor(data['obs'], dtype=torch.float32)
        act = torch.as_tensor(data['act'], dtype=torch.float32)
        ret = torch.as_tensor(data['ret'], dtype=torch.float32).unsqueeze(1)
        adv = torch.as_tensor(data['adv'], dtype=torch.float32).unsqueeze(1)
        old_logp = torch.as_tensor(data['logp'], dtype=torch.float32).unsqueeze(1)
        
        # Policy loss
        for _ in range(self.train_iters):
            # Get new action distribution
            action_mean, _, _ = self.policy_net(obs.unsqueeze(1))
            action_std = torch.clamp(self.action_std, 0.1, 1.0)
            dist = Normal(action_mean, action_std)
            
            # New log probability
            new_logp = dist.log_prob(act).sum(dim=1, keepdim=True)
            
            # Ratio
            ratio = torch.exp(new_logp - old_logp)
            
            # Clipped surrogate objective
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            policy_loss = -torch.min(ratio * adv, clip_adv).mean()
            
            # Value loss
            value_pred = self.value_net(obs)
            value_loss = ((value_pred - ret) ** 2).mean()
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # KL divergence early stopping
            with torch.no_grad():
                approx_kl = (old_logp - new_logp).mean()
                if approx_kl > 1.5 * self.target_kl:
                    break
            
            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()

# ============================================================================
# 5. TRAINING LOOP & VISUALIZATION
# ============================================================================

def train_plasma_controller(num_episodes=100, steps_per_episode=1000):
    """Main training loop"""
    
    print("=" * 60)
    print("TRAINING PLASMA ML CONTROLLER")
    print("=" * 60)
    
    # Initialize environment and agent
    env = PlasmaRLEnvironment()
    agent = PPOAgent(obs_dim=512, act_dim=4)
    buffer = PPOBuffer(obs_dim=512, act_dim=4, size=steps_per_episode)
    
    # Training metrics
    episode_rewards = []
    dust_mass_history = []
    hotspot_history = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        hidden_state = None
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < steps_per_episode:
            # Get action from agent
            action, value, logp, hidden_state = agent.get_action(obs, hidden_state)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store in buffer
            buffer.store(obs, action, reward, value, logp)
            
            # Update
            obs = next_obs
            episode_reward += reward
            step += 1
            
            if done:
                # Finish path with last value estimate
                last_val = 0 if done else agent.get_action(obs, hidden_state)[1]
                buffer.finish_path(last_val)
                
        # Update agent
        if buffer.ptr == buffer.max_size:
            agent.update(buffer)
            
        # Record metrics
        episode_rewards.append(episode_reward)
        dust_mass_history.append(info['dust_mass'])
        hotspot_history.append(info['hotspot_count'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:3d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Dust Mass: {info['dust_mass']:.3f} | "
                  f"Hotspots: {info['hotspot_count']:2d} | "
                  f"RF Power: {info['rf_power']:.2f} kW")
    
    # Plot training results
    plot_training_results(episode_rewards, dust_mass_history, hotspot_history)
    
    return agent, env

def plot_training_results(rewards, dust_mass, hotspots):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dust mass preservation
    axes[0, 1].plot(dust_mass)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Safe threshold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Dust Mass (normalized)')
    axes[0, 1].set_title('Dust Preservation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hot spot frequency
    axes[1, 0].plot(hotspots)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Hot Spots Detected')
    axes[1, 0].set_title('Hot Spot Formation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 1].hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plasma_ml_training.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# 6. REAL-TIME CONTROL SYSTEM
# ============================================================================

class RealTimePlasmaController:
    """Real-time control system for deployment"""
    
    def __init__(self, model_path=None):
        # Load or create model
        self.model = PlasmaControlNet(input_dim=512, control_dim=4)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(feature_dim=512)
        
        # Sensor buffer
        self.sensor_buffer = deque(maxlen=1000)
        
        # Control history
        self.control_history = deque(maxlen=100)
        
        # Safety monitor
        self.safety_monitor = SafetyLayer()
        
        # System state
        self.system_state = {
            'T_dust': 300.0,
            'Gamma': 100.0,
            'vacuum': 1e-6,
            'rf_power': 0.0,
            'pulse_width': 50.0
        }
        
        # Performance metrics
        self.metrics = {
            'avg_reward': 0.0,
            'dust_survival': 1.0,
            'hotspot_rate': 0.0,
            'control_stability': 0.0
        }
        
    def update_sensors(self, sensor_readings: Dict):
        """Update sensor buffer with new readings"""
        self.sensor_buffer.append(sensor_readings)
        
        # Update system state from sensors
        if 'T_electron' in sensor_readings:
            # Simplified dust temperature estimate
            self.system_state['T_dust'] = 300.0 + 0.05 * sensor_readings['T_electron']
            
        if 'B_field' in sensor_readings and 'plasma_density' in sensor_readings:
            # Simplified Gamma estimate
            b_field = abs(sensor_readings['B_field'])
            density = sensor_readings['plasma_density']
            if density > 1e15:
                self.system_state['Gamma'] = 100.0 + 50.0 * (b_field / density)
                
    def get_control_action(self) -> Dict:
        """Get next control action based on current state"""
        if len(self.sensor_buffer) < 100:
            # Return safe baseline if not enough data
            return self._get_baseline_controls()
        
        # Extract features
        current_readings = self.sensor_buffer[-1]
        features = self.feature_extractor.extract_features(
            current_readings, 
            list(self.sensor_buffer)
        )
        
        # Convert to tensor
        obs_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Get control action from neural network
        with torch.no_grad():
            controls, attn_weights, _ = self.model(obs_t, state_info=self.system_state)
            controls = controls.squeeze(0).numpy()
            
        # Decode controls
        control_action = {
            'rf_power': float(np.clip(controls[0], 0.0, 2.0)),
            'pulse_width': float(np.clip(controls[1], 10.0, 100.0)),
            'mod_freq': float(np.clip(controls[2], 7.73, 7.93)),
            'phase': float(controls[3] % 360.0)
        }
        
        # Store in history
        self.control_history.append(control_action)
        
        # Update system state
        self.system_state['rf_power'] = control_action['rf_power']
        self.system_state['pulse_width'] = control_action['pulse_width']
        
        return control_action
    
    def _get_baseline_controls(self) -> Dict:
        """Return safe baseline controls"""
        return {
            'rf_power': 0.5,      # 500 W
            'pulse_width': 50.0,  # 50 μs
            'mod_freq': 7.83,     # Schumann resonance
            'phase': 180.0        # 180 degrees
        }
    
    def emergency_stop(self) -> Dict:
        """Emergency shutdown sequence"""
        print("EMERGENCY STOP ACTIVATED!")
        return {
            'rf_power': 0.0,
            'pulse_width': 0.0,
            'mod_freq': 7.83,
            'phase': 0.0,
            'emergency': True
        }
    
    def get_system_status(self) -> Dict:
        """Get current system status report"""
        status = {
            'system_state': self.system_state.copy(),
            'metrics': self.metrics.copy(),
            'sensor_count': len(self.sensor_buffer),
            'control_history_length': len(self.control_history),
            'timestamp': time.time()
        }
        
        # Calculate control stability
        if len(self.control_history) > 10:
            rf_powers = [c['rf_power'] for c in list(self.control_history)[-10:]]
            self.metrics['control_stability'] = 1.0 / (1.0 + np.std(rf_powers))
            
        return status

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("PLASMA ML CONTROL SYSTEM - v1.0")
    print("=" * 60)
    
    # Option 1: Train new controller
    train_new = input("Train new controller? (y/n): ").lower().strip() == 'y'
    
    if train_new:
        print("\nStarting training...")
        agent, env = train_plasma_controller(num_episodes=50, steps_per_episode=500)
        
        # Save trained model
        torch.save(agent.policy_net.state_dict(), 'plasma_control_model.pt')
        print("Model saved to 'plasma_control_model.pt'")
        
    else:
        # Option 2: Run real-time simulation
        print("\nStarting real-time simulation...")
        controller = RealTimePlasmaController(model_path='plasma_control_model.pt')
        
        # Simulate sensor data
        sensor_sim = PlasmaSensorSimulator()
        
        # Run simulation for 1000 steps
        for step in range(1000):
            # Get sensor readings
            sensor_readings = sensor_sim.read_sensors(controller.system_state)
            
            # Update controller
            controller.update_sensors(sensor_readings)
            
            # Get control action
            controls = controller.get_control_action()
            
            # Print status every 100 steps
            if step % 100 == 0:
                status = controller.get_system_status()
                print(f"\nStep {step}:")
                print(f"  RF Power: {controls['rf_power']:.2f} kW")
                print(f"  Pulse Width: {controls['pulse_width']:.1f} μs")
                print(f"  Dust Temp: {status['system_state']['T_dust']:.1f} K")
                print(f"  Gamma: {status['system_state']['Gamma']:.1f}")
                
            # Emergency stop check
            if controller.system_state['T_dust'] > 2500:
                print("\n⚠️  DUST OVERHEAT DETECTED!")
                controls = controller.emergency_stop()
                break
                
        print("\nSimulation complete!")
        
    print("\n" + "=" * 60)
    print("System ready for experimental deployment")
    print("=" * 60)

if __name__ == "__main__":
    main()
