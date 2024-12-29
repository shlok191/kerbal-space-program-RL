from base_environment import KSPEnvironment, MissionStatus
from typing import Dict
import numpy as np
    
class KerbinOrbitalEnvironment(KSPEnvironment):
    
    def __init__(self, ip_address: str, rpc_port: int, stream_port: int, 
        connection_name: str, vessel_name: str, max_timesteps: int,
        target_apoasis: int, target_periapsis: int, max_altitude: int,
        max_velocity: int):
        """_summary_

        Args:
            ip_address (str): The IP address to connect to
            rpc_port (int): Handles updates to the active vehicle (throttle, pitch, yaw etc.)
            stream_port (int): Handles real-time updates from the environment (live telemetery updates)
            connection_name (str): The name of the connection -- Assists with debugging!
            vessel_name (str): The name of the vessel to control -- Also assists with debugging!
            max_timesteps (int): The maximum number of timesteps to run the simulation for
            target_apoasis (int): The target apoasis for the orbit
            target_periapsis (int): The target periapsis for the orbit
            max_altitude (int): The maximum altitude for the vessel
            max_velocity (int): The maximum achievable velocity for the vessel
        """
        
        super().__init__(ip_address, rpc_port, stream_port, connection_name, vessel_name, max_timesteps)
        
        # Defining orbital specific parameters
        # Unit: Standard International (SI) Units :)

        self.target_apoapsis = target_apoasis
        self.target_periapsis = target_periapsis
        self.max_altitude = max_altitude
    
        # Defining reward parameters
        self.previous_state = None
        self.cumulative_reward = None
    
        
    def get_normalized_vessel_state(self) -> Dict[str, float]:
        """Gets the vessel states in a normalized format

        Returns:
            Dict[str, float]: A dictionary containing state values
        """
        
        # Fetching the  orbital parameters
        raw_updates = self.get_vessel_updates()
        
        # Fetching normalized states
        raw_updates.update({
            "altitude": raw_updates['altitude'] / self.max_altitude,
            "velocity": raw_updates['velocity'] / self.max_velocity,
            "pitch": raw_updates['pitch'] / 180,
            "yaw": raw_updates['yaw'] / 180,
            "roll": raw_updates['roll'] / 180,
            "apoapsis": raw_updates['apoapsis'] / self.target_apoapsis,
            "periapsis": raw_updates['periapsis'] / self.target_periapsis,
            "thrust": raw_updates["thrust"] / raw_updates["max_thrust"],
            "fuel": raw_updates['fuel'] / raw_updates['max_fuel'],
            "oxidizer": raw_updates['oxidizer'] / raw_updates['max_oxidizer'],
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
            self.vessel.orbit.periapsis_altitude > target.atmosphere_depth and 
            self.vessel.orbit.eccentricity < 1):
            
            return True
    
    def check_terminal_state(self):
        """Returns the state of the episode

        Returns:
            Enum: MissionStatus Categories
        """
        
        # Checks if the mission is complete!
        if self.check_orbit():
            return MissionStatus.COMPLETED
        
        # Returns the default terminal state
        return super().check_terminal_state()
    
    def step(self, controls: Dict[str, float]) -> float:
        """Performs a step in the environment and returns a reward

        Args:
            controls (Dict[str, float]): The controls to be updated
            
        Returns:
            float: The reward for the current state
        """
        
        super().step(controls)
        