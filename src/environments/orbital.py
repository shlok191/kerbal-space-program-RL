from base import KSPEnvironment, MissionStatus
from typing import Dict
import numpy as np

KERBIN_RADIUS = 598725.62

class KerbinOrbitalEnvironment(KSPEnvironment):
    
    def __init__(self, 
            ip_address: str,
            rpc_port: int, 
            stream_port: int, 
            connection_name: str,
            vessel_name: str,
            rcs_enabled: bool,
            max_steps: int,
            target_apoapsis: int,
            target_periapsis: int, 
            max_altitude: int = 100_000,
            max_velocity: int = 5_000,
            step_sim_time: int = 3):
        """Defines a custom environment for the Kerbin Orbital Mission!

        Args:
            ip_address (str): The IP address to connect to
            rpc_port (int): Handles updates to the active vehicle (throttle, pitch, yaw etc.)
            stream_port (int): Handles real-time updates from the environment (live telemetery updates)
            connection_name (str): The name of the connection -- Assists with debugging!
            vessel_name (str): The name of the vessel to control -- Also assists with debugging!
            RCS_enabled (bool): Whether or not to enable the Reaction Control System
            max_steps (int): The maximum number of steps to run the simulation for
            target_apoapsis (int): The target apoasis for the orbit
            target_periapsis (int): The target periapsis for the orbit
            max_altitude (int): The maximum altitude for the vessel
            max_velocity (int): The maximum velocity for the vessel
            step_sim_time (int): The time to step the simulation by
        """
        
        super().__init__(ip_address, rpc_port, stream_port, connection_name, vessel_name, max_steps, rcs_enabled, step_sim_time)
        
        # Defining orbital specific parameters
        # Unit: Standard International (SI) Units :)

        self.target_apoapsis = target_apoapsis
        self.target_periapsis = target_periapsis
        self.max_altitude = max_altitude
        self.max_velocity = max_velocity
        
        # Defining reward parameters
        self.previous_state = None
        self.cumulative_reward = None
    
        self.action_space = {
            "throttle": 0, # Ranges from [0, 1]
            "pitch": 0,    # Ranges from [0, 1]
            "yaw": 0,      # Ranges from [0, 1]
            "roll": 0      # Ranges from [0, 1]
        }    
        
        # Defines the ideal pitch across the orbital maneuver stages!
        self.ideal_pitch = {
            "stage_1": 0,
            "stage_2": 0.25,
            "stage_3": 0.5,
        }
        
    def get_normalized_vessel_state(self) -> Dict[str, float]:
        """Gets the vessel states in a normalized format

        Returns:
            Dict[str, float]: A dictionary containing state values
        """
        
        eps = 1e-8
        
        # Fetching the  orbital parameters
        raw_updates = self.get_vessel_updates()
        
        # Fetching normalized states
        raw_updates.update({
            
            # Updating reference related values
            "altitude": raw_updates['altitude'] / (self.max_altitude + eps),
            "velocity": [v / (self.max_velocity + eps) for v in raw_updates['velocity']],
            
            # Updating vessel orientation values
            "pitch": raw_updates['pitch'] / 90,
            "yaw": raw_updates['yaw'] / 180,
            "roll": raw_updates['roll'] / 180,
            
            # Updating the orbit related values
            "apoapsis": raw_updates['apoapsis'] / (self.target_apoapsis + eps),
            "periapsis": raw_updates['periapsis'] / (self.target_periapsis + eps),
            
            # Updating fuel related values
            "fuel": raw_updates['fuel'] / (raw_updates['max_fuel'] + eps),
            "oxidizer": raw_updates['oxidizer'] / (raw_updates['max_oxidizer'] + eps),

            # Updating the magic variable
            "thrust": raw_updates["thrust"] / (raw_updates["avail_thrust"] + eps),
        })
        
        return raw_updates
    
    def check_orbit(self) -> bool:
        """Checks if the vessel has achieved orbit around Kerbin!
        
        Args:
            target_name (str): The name of the target to achieve orbit around
        
        Returns:
            bool: Returns True if the vessel has achieved orbit, False otherwise
        """
        
        # Defining the target object
        target = self.connection.space_center.bodies["Kerbin"]
        
        if (self.vessel.orbit.body == target and 
            self.vessel.orbit.periapsis_altitude >= target.atmosphere_depth and 
            self.vessel.orbit.eccentricity <= 1):
            
            return True

        return False
    
    def check_terminal_state(self):
        """Returns the state of the episode

        Returns:
            Enum: MissionStatus Categories
        """
        
        # Checks if the mission is complete!
        if self.check_orbit():
            return MissionStatus.COMPLETED
        
        if self.vessel.flight().mean_altitude > self.max_altitude:
            return MissionStatus.OUT_OF_BOUNDS
        
        # Returns the default terminal state
        return super().check_terminal_state()
    
    def step(self, controls: Dict[str, float]) -> float:
        """Performs a step in the environment and returns a reward

        Reward Modeling Strategy:
        --------------------------------------------------
            
            1. Progressive rewards for optimal trajectory
            2. Penalizes inefficient fuel usage
            3. Encouraging gradual gravity turn
            4. Rewarding proximity to target orbital parameters
            5. Heavily penalizing mission failure scenarios
            
        --------------------------------------------------
        
        Args:
            controls (Dict[str, float]): The controls to be updated
            
        Returns:
            float: The reward for the current state
        """

        # Performing the first initial step
        super().step(controls)
        
        # Defining our initial reward
        reward = 0
        
        # Fetching relevant state values
        current_state = self.get_normalized_vessel_state()
        
        altitude = current_state['altitude']
        velocity = current_state['velocity']
        pitch = current_state['pitch']
        periapsis_altitude = current_state['periapsis']
        apoapsis_altitude = current_state['apoapsis']
                
        # Phase 1: Initial Launch Phase -- 0 KM to 10 KM
        if altitude < 0.1:
            
            # This matches Kerbin's natural tilt when normalized :)
            ideal_pitch = 0.17
            
            # Rewarding vertical and penalizing horizontal velocity
            reward -= 5 * (velocity[0])
            reward += 10 * velocity[1]
            
            # Pitch alignment bonus for staying close to the ideal pitch
            pitch_alignment = 1 - abs(pitch - ideal_pitch)
            reward += 5 * pitch_alignment
            
            
        # Phase 2: Early Gravity Turn Phase -- 10 KM to 30 KM
        elif 0.1 <= altitude < 0.3:
            
            # Defines a progressive pitch change based on current altitude
            ideal_pitch = 0.17 + 0.33 * ((altitude - 0.1) / 0.2)
            
            # Pitch alignment bonus
            pitch_alignment = 1 - abs(pitch - ideal_pitch)
        
        
        # Phase 3: Late Gravity Turn Phase -- 20 KM to 100 KM
        elif altitude >= 0.3:
            
            # This matches 45 degrees, which is what I believe we should target!
            ideal_pitch = 0.5 
        
            pitch_alignment = 1 - abs(pitch - ideal_pitch)
            reward += 10 * pitch_alignment
        
            # Starting to give out rewards for meeting apoapsis and periapsis targets
            apoapsis_proximity = 1 - abs(apoapsis_altitude)
            periapsis_proximity = 1 - abs(periapsis_altitude)
            
            reward += 15 * apoapsis_proximity
            reward += 15 * periapsis_proximity
        
            
        # Defining a fuel efficiency penalty
        current_fuel = self.vessel.resources.amount('LiquidFuel')
        fuel_used = self.previous_fuel - current_fuel
        reward -= 3 * fuel_used
        
        self.previous_fuel = current_fuel
        
        # Also devising a time penalty
        reward -= 0.2 * self.elapsed_time
        
        # Fetching the episode state
        episode_state = self.check_terminal_state()
        
        mission_status = None
        
        # Penalizing negative scenarios that end the episode!
        if episode_state == MissionStatus.CRASHED:
            reward -= 500
            mission_status = MissionStatus.CRASHED
            
        elif episode_state == MissionStatus.OUT_OF_BOUNDS:
            reward -= 500
            mission_status = MissionStatus.OUT_OF_BOUNDS
            
        elif episode_state == MissionStatus.OUT_OF_FUEL:
            reward -= 200
            mission_status = MissionStatus.OUT_OF_FUEL
            
        elif episode_state == MissionStatus.OUT_OF_TIME:
            reward -= 200
            mission_status = MissionStatus.OUT_OF_TIME
             
        # Best case scenario, really hoping we get here!
        elif episode_state == MissionStatus.COMPLETED:
            reward += 1000
            mission_status = MissionStatus.COMPLETED
           
        return reward, mission_status