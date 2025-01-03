# Creates a custom Python API to connect with and run a KSP environment
import krpc
from typing import Dict
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
            rcs_enabled: bool,
            step_sim_time: int = 10):
        """Creates a connection with kRPC to remote server along given ports!

        Args:
            ip_address (str): The IP address to connect to
            rpc_port (int): Handles updates to the active vehicle (throttle, pitch, yaw etc.)
            stream_port (int): Handles real-time updates from the environment (live telemetery updates)
            connection_name (str): The name of the connection -- Assists with debugging!
            vessel_name (str): The name of the vessel to control -- Also assists with debugging!
            max_steps (int): The maximum number of steps to run the simulation for
            rcs_enabled (bool): Whether or not the reaction control system is enabled
            step_sim_time (int): The time to step the simulation by
        """

        # Storing object variables
        self.ip_address = ip_address
        self.rpc_port = rpc_port
        self.stream_port = stream_port
        self.connection_name = connection_name
        self.vessel_name = vessel_name
        self.rcs_enabled = rcs_enabled
        
        # Defining time tracking variables
        self.elapsed_time = 0
        self.step_sim_time = step_sim_time
        
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
                
        # Getting the starting time in KSP time
        self.start_time = self.connection.space_center.ut
        
        # Defining the maximum number of steps!
        self.max_steps = max_steps    
                    
        # Introducing the speedup for faster training :)
        self.connection.space_center.physics_warp_factor = 2
        
        ref_frame = self.vessel.orbit.body.reference_frame
        
        # Defining the flight and vessel stream that collects info through a TCP/IP connection
        self.flight_stream = self.connection.add_stream(self.vessel.flight, ref_frame)
        self.vessel_stream = self.connection.add_stream(getattr, self.connection.space_center, 'active_vessel')
        
    def get_vessel_updates(self) -> Dict[str, float]:
        """Returns important telemetery information regarding the vessel
        
        Returns:
            Dict[str, float]: Contains atributes and their values (Altitude, Velocity, etc.)
        """
    
        try:
            # Collecting flight info
            flight = self.flight_stream()
            vessel = self.vessel_stream()
        
        except:
            
            print("Encountered an error while fetching vessel updates!")
            self.older_state['altitude'] = -100
            return self.older_state
        
        # Defining the dictionary to return with critical vessel state parameters
        vessel_state = {
            
            # Flight Telemetry
            "altitude": flight.mean_altitude,   # Altitude above sea level [meters]
            "velocity": flight.velocity,        # Velocity vector [m/s]
            
            # Orientation
            "pitch": flight.pitch,         # Pitch angle [degrees] 
            "yaw": flight.sideslip_angle,  # Yaw angle [degrees]
            "roll": flight.roll,           # Roll angle [degrees]
            
            # Thrust and Engine Metrics
            "thrust": vessel.thrust,                  # Current thrust output [Newtons]
            "avail_thrust": vessel.available_thrust,  # Available thrust [Newtons]

            # Orbital Parameters
            "apoapsis": vessel.orbit.apoapsis_altitude,             # Highest point in orbit [meters]
            "periapsis": vessel.orbit.periapsis_altitude,           # Lowest point in orbit [meters]
            
            # Mass and Resources
            "mass": vessel.mass,            # Total mass [kg]
            "dry_mass": vessel.dry_mass,    # Dry mass [kg]
            "fuel": vessel.resources.amount('LiquidFuel'),      # Liquid fuel remaining [units]
            "max_fuel": vessel.resources.max('LiquidFuel'),     # Max liquid fuel capacity [units]
            "oxidizer": vessel.resources.amount('Oxidizer'),    # Oxidizer remaining [units]
            "max_oxidizer": vessel.resources.max('Oxidizer'),   # Max oxidizer capacity [units]
            "electric_charge": vessel.resources.amount('ElectricCharge'),  # Electric charge [units]
            
            # Control State
            "throttle": vessel.control.throttle,  # Current throttle setting [0.0 - 1.0]
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
        
        self.older_state = vessel_state
        
        return vessel_state

    def update_vehicle_controls(self, controls: Dict[str, float]):
        """Updates the vehicle controls based on the given dictionary
        
        Args:
            controls (Dict[str, float]): Contains the control values to update (Throttle, Pitch, Yaw etc.)
        """
        
        def clamp(value, min_val=0.0, max_val=1.0):
            return max(min_val, min(value.item(), max_val))
        
        # Updates the vehicle controls
        self.vessel.control.throttle = clamp(controls.get("throttle", self.vessel.control.throttle))
        self.vessel.control.pitch = clamp(controls.get("pitch", self.vessel.control.pitch))
        self.vessel.control.yaw = clamp(controls.get("yaw", self.vessel.control.yaw))
        self.vessel.control.roll = clamp(controls.get("roll", self.vessel.control.roll))
        
        # Adding conditional RCS controls
        if self.rcs_enabled:
            
            self.vessel.control.up = clamp(controls.get("rcs_up", self.vessel.control.up))
            self.vessel.control.down = clamp(controls.get("rcs_down", self.vessel.control.down))
            self.vessel.control.left = clamp(controls.get("rcs_left", self.vessel.control.left))
            self.vessel.control.right = clamp(controls.get("rcs_right", self.vessel.control.right))
            self.vessel.control.forward = clamp(controls.get("rcs_forward", self.vessel.control.forward))
            self.vessel.control.backward = clamp(controls.get("rcs_backward", self.vessel.control.backward))
            
    def step(self, controls: Dict[str, float]):
        """
        Advances the simulation by a specified duration.
        
        Parameters:
        conn (krpc.Connection): The active kRPC connection.
        """
        
        # Unpausing the game (custom state)
        # self.connection.krpc.paused = False

        # Recording the target unpausing time
        target_ut = self.connection.space_center.ut + self.step_sim_time

        # Updating the vehicle controls
        self.update_vehicle_controls(controls)
        
        # Wait until the target UT is reached
        while self.connection.space_center.ut < target_ut:
            time.sleep(0.0001)
        
        # Updating the elapsed time!
        self.elapsed_time += self.step_sim_time
        
    def restart_episode(self):
        """
        Restarts the episode by reverting the simulation to the initial state.
        
        Parameters:
        conn (krpc.Connection): The active kRPC connection.
        """
        
        # Reverting to launch!
        self.connection.space_center.quickload()
        
        time.sleep(1.5)
        
        self.vessel = self.connection.space_center.active_vessel
        self.vessel.name = self.vessel_name
        
        # Getting the starting time in KSP time
        self.start_time = self.connection.space_center.ut
        self.elapsed_time = 0
        
        # Introducing the speedup for faster training :)
        self.connection.space_center.physics_warp_factor = 1.5
        
        ref_frame = self.vessel.orbit.body.reference_frame
        
        # Defining the flight and vessel stream that collects info through a TCP/IP connection
        self.flight_stream = self.connection.add_stream(self.vessel.flight, ref_frame)
        self.vessel_stream = self.connection.add_stream(getattr, self.connection.space_center, 'active_vessel')

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
    
    def check_terminal_state(self, altitude: float) -> str:
        """Checks if the a terminal state has been reached
        
        Returns:
            str: Returns the terminal state
        """
        
        # Checking if the vessel has run out of fuel
        if self.vessel.resources.amount("LiquidFuel") == 0 or self.vessel.resources.amount("Oxidizer") == 0:
            return MissionStatus.OUT_OF_FUEL
        
        # Checking if the vessel has run out of time
        if self.connection.space_center.ut > self.start_time + self.max_steps * self.step_sim_time:
            return MissionStatus.OUT_OF_TIME
        
        # The default state :)
        return MissionStatus.IN_PROGRESS