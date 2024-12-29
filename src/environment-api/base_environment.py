# Creates a custom Python API to connect with and run a KSP environment
import krpc
from typing import List, Dict
import time
from enum import Enum
    
class MissionStatus(Enum):
    IN_PROGRESS = 0
    CRASHED = 1
    ACHIEVED_ORBIT = 2
    OUT_OF_FUEL = 3
    OUT_OF_BOUNDS = 4
    OUT_OF_TIME = 5
    COMPLETED = 6
    
class KSPEnvironment():
    
    def __init__(self, 
            ip_address: str,
            rpc_port: int, 
            stream_port: int,
            connection_name: str,
            vessel_name: str,
            max_steps: int,
            rcs_enabled: bool):
        """Creates a connection with kRPC to remote server along given ports!

        Args:
            ip_address (str): The IP address to connect to
            rpc_port (int): Handles updates to the active vehicle (throttle, pitch, yaw etc.)
            stream_port (int): Handles real-time updates from the environment (live telemetery updates)
            connection_name (str): The name of the connection -- Assists with debugging!
            vessel_name (str): The name of the vessel to control -- Also assists with debugging!
            max_steps (int): The maximum number of steps to run the simulation for
            rcs_enabled (bool): Whether or not the reaction control system is enabled
        """
        
        # Storing object variables
        self.ip_address = ip_address
        self.rpc_port = rpc_port
        self.stream_port = stream_port
        self.connection_name = connection_name
        self.vessel_name = vessel_name
        self.rcs_enabled = rcs_enabled
        
        # Attempting to create a connection with the server
        try:
            
            self.connection = krpc.connect(
                name=connection_name,
                address=ip_address,
                rpc_port = rpc_port,
                stream_port=stream_port
            )   

            # Ensuring the connection is valid
            print(self.connection.krpc.get_status())
            
            self.vessel = self.connection.space_center.active_vessel
            self.vessel.name = vessel_name
            
        except:
            raise ConnectionError("Could not establish a connection!")
                
        # Defining initial control settings
        
        self.controls = {
            "throttle": 0, # Let Throttle range from [0, 1]
            "pitch": 0,    # Let Pitch range from [0, 1]
            "yaw": 0,      # Let Yaw range from [0, 1] 
            "roll": 0      # Let Roll range from [0, 1]
        }
        
        # Getting the starting time in KSP time
        self.start_time = self.connection.space_center.ut
        
        # Defining the maximum number of steps!
        self.max_steps = max_steps    

        # Starting with a clean state :)        
        self.restart_episode()

    def get_vessel_updates(self) -> Dict[str, float]:
        """Returns important telemetery information regarding the vessel
        
        Returns:
            Dict[str, float]: Contains atributes and their values (Altitude, Velocity, Acceleration etc.)
        """
        
       # Defining the dictionary to return with critical vessel state parameters
        vessel_state = {
            
            # Flight Telemetry
            "altitude": self.vessel.flight().mean_altitude,  # Altitude above sea level [meters]
            "surface_altitude": self.vessel.flight().surface_altitude,  # Altitude above terrain [meters]
            "velocity": self.vessel.flight().velocity,  # Velocity vector [m/s]
            "acceleration": self.vessel.flight().acceleration,  # Acceleration vector [m/s²]
            
            # Orientation
            "pitch": self.vessel.flight().pitch,  # Pitch angle [degrees]
            "yaw": self.vessel.flight().yaw,  # Yaw angle [degrees]
            "roll": self.vessel.flight().roll,  # Roll angle [degrees]
            
            # Thrust and Engine Metrics
            "thrust": self.vessel.thrust(),  # Current thrust output [Newtons]
            "avail_thrust": self.vessel.available_thrust(),  # Available thrust [Newtons]
            "max_thrust": self.vessel.max_thrust(),  # Maximum thrust at current conditions [Newtons]
            "vacuum_specific_impulse": self.vessel.vacuum_specific_impulse(),  # ISP in vacuum [seconds]
            "specific_impulse": self.vessel.specific_impulse(),  # Current ISP [seconds]
            
            # Orbital Parameters
            "apoapsis": self.vessel.orbit.apoapsis_altitude,  # Highest point in orbit [meters]
            "periapsis": self.vessel.orbit.periapsis_altitude,  # Lowest point in orbit [meters]
            "time_to_apoapsis": self.vessel.orbit.time_to_apoapsis,  # Time until apoapsis [seconds]
            "time_to_periapsis": self.vessel.orbit.time_to_periapsis,  # Time until periapsis [seconds]
            "eccentricity": self.vessel.orbit.eccentricity,  # Orbital eccentricity (unitless)
            "inclination": self.vessel.orbit.inclination,  # Orbital inclination [degrees]
            
            # Mass and Resources
            "mass": self.vessel.mass(),  # Total mass [kg]
            "dry_mass": self.vessel.dry_mass(),  # Dry mass [kg]
            "fuel": self.vessel.resources.amount('LiquidFuel'),  # Liquid fuel remaining [units]
            "max_fuel": self.vessel.resources.max('LiquidFuel'),  # Max liquid fuel capacity [units]
            "oxidizer": self.vessel.resources.amount('Oxidizer'),  # Oxidizer remaining [units]
            "max_oxidier": self.vessel.resources.max('Oxidizer'),  # Max oxidizer capacity [units]
            "electric_charge": self.vessel.resources.amount('ElectricCharge'),  # Electric charge [units]
            
            # Control State
            "throttle": self.vessel.control.throttle,  # Current throttle setting [0.0 - 1.0]
        }

        # Conditional RCS State
        if self.rcs_enabled:
            
            vessel_state.update({
                
                # RCS State
                "available_rcs_torque": self.vessel.available_rcs_torque(),  # Max RCS torque [Nm]
                "available_rcs_force": self.vessel.available_rcs_force(),  # Max RCS force [N]
                "monopropellant": self.vessel.resources.amount('MonoPropellant'),  # Remaining monopropellant [units]
                "monopropellant_capacity": self.vessel.resources.max('MonoPropellant'),  # Max monopropellant capacity [units]
            })
            
        return vessel_state

    def update_vehicle_controls(self, controls: Dict[str, float]):
        """Updates the vehicle controls based on the given dictionary
        
        Args:
            controls (Dict[str, float]): Contains the control values to update (Throttle, Pitch, Yaw etc.)
        """
        
        # Updates the vehicle controls
        self.vessel.control.throttle = controls.get("throttle", self.vessel.control.throttle)
        self.vessel.control.pitch = controls.get("pitch", self.vessel.control.pitch)
        self.vessel.control.yaw = controls.get("yaw", self.vessel.control.yaw)
        self.vessel.control.roll = controls.get("roll", self.vessel.control.roll)
        
        # Adding conditional RCS controls
        if self.rcs_enabled:
            
            self.vessel.control.up = controls.get("rcs_up", self.vessel.control.up)
            self.vessel.control.down = controls.get("rcs_down", self.vessel.control.down)
            self.vessel.control.left = controls.get("rcs_left", self.vessel.control.left)
            self.vessel.control.right = controls.get("rcs_right", self.vessel.control.right)
            self.vessel.control.forward = controls.get("rcs_forward", self.vessel.control.forward)
            self.vessel.control.backward = controls.get("rcs_backward", self.vessel.control.backward)
            
    def step_simulation(self, controls: Dict[str, float], step_duration: int):
        """
        Advances the simulation by a specified duration.
        
        Parameters:
        conn (krpc.Connection): The active kRPC connection.
        step_duration (float): Duration to advance the simulation, in seconds.
        """
        
        # Unpausing the game (custom state)
        self.connection.krpc.paused = False

        # Recording the target unpausing time
        target_ut = self.connection.space_center.ut + step_duration

        # Updating the vehicle controls
        self.update_vehicle_controls(controls)
        
        # Wait until the target UT is reached
        while self.connection.space_center.ut < target_ut:
            time.sleep(0.01)

        # Pause the game (default state)
        self.connection.krpc.paused = True
        
    def restart_episode(self):
        """
        Restarts the episode by reverting the simulation to the initial state.
        
        Parameters:
        conn (krpc.Connection): The active kRPC connection.
        """
        
        # Reverting the simulation to the initial state
        self.connection.space_center.quickload()

        # Unpausing the game (default state)
        self.connection.krpc.paused = True
        
    def get_distances(self, target_name: str) -> Dict[str, float]:
        """Returns the distances between the vessel and the target
        
        Args:
            target_name (str): The name of the target to calculate distances from
        
        Returns:
            Dict[str, float]: Contains the distances between the vessel and the target
        """
        
        # Defining the target object
        target = self.connection.space_center.bodies[target_name]
        
        # Defining the dictionary to return
        distances = {
            "distance": self.vessel.orbit.body.surface_position(target.reference_frame),
            "distance_to_target": self.vessel.orbit.body.surface_position(target.reference_frame).distance
        }
        
        return distances
    
    def check_terminal_state(self) -> str:
        """Checks if the a terminal state has been reached
        
        Returns:
            str: Returns the terminal state
        """

        # Since there is no direct crash mechanism, checking if the Kerbins are alive and well!
        for kerbin in self.vessel.crew:
            
            if kerbin.roster_status in ['dead', 'missing']:
                return MissionStatus.CRASHED
        
        # Checking if the vessel has run out of fuel
        if self.vessel.resources.amount("LiquidFuel") == 0 or self.vessel.resources.amount("Oxidizer") == 0:
            return MissionStatus.OUT_OF_FUEL
        
        # Checking if the vessel is out of bounds
        if self.vessel.orbit.body.surface_altitude < 0:
            return MissionStatus.OUT_OF_BOUNDS
        
        # Checking if the vessel has run out of time
        if self.connection.space_center.ut > self.start_time + self.max_timesteps:
            return MissionStatus.OUT_OF_TIME
        
        # The default state :)
        return MissionStatus.IN_PROGRESS