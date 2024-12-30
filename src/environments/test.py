import time
import logging
from typing import Dict, Any

def test_ksp_connection(
    ip_address: str = "127.0.0.1",
    rpc_port: int = 50000,
    stream_port: int = 50001,
    max_steps: int = 1000,
    log_level: int = logging.INFO
) -> None:
    """
    Comprehensive test of KSP connection, displaying telemetry and normalized states.
    
    This function connects to a Kerbin Orbital Environment and performs the following:
    1. Establishes a connection to the KSP simulation
    2. Retrieves and logs raw vessel telemetry data
    3. Retrieves and logs normalized vessel states
    4. Provides error handling and logging
    
    Args:
        ip_address (str): IP address of the KSP connection. Defaults to localhost.
        rpc_port (int): RPC port for connection. Defaults to 50000.
        stream_port (int): Stream port for connection. Defaults to 50001.
        max_steps (int): Maximum number of simulation steps to run. Defaults to 1000.
        log_level (int): Logging level. Defaults to logging.INFO.
    
    Raises:
        Exception: If there are issues with connection or data retrieval
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    try:
        # Import here to allow for potential import errors to be caught
        from orbital import KerbinOrbitalEnvironment

        # Create the orbital environment connection
        logger.info("Initializing KSP Orbital Environment connection...")
        base_env = KerbinOrbitalEnvironment(
            ip_address=ip_address,
            rpc_port=rpc_port,
            stream_port=stream_port,
            connection_name="Test_Connection",
            vessel_name="Test_Vessel",
            max_steps=max_steps,
            rcs_enabled=False,
            step_sim_time=1,
            target_apoapsis=1_000_000,
            target_periapsis=1_000_000
        )
        logger.info("Connection established successfully.")

        # Run simulation steps
        for step in range(max_steps):
            try:
                # Log step information
                logger.info(f"\n{'='*50}\nStep {step + 1}:\n{'='*50}")

                # Retrieve and log raw vessel state
                raw_state = _log_raw_vessel_state(base_env.get_vessel_updates())

                # Retrieve and log normalized vessel state
                try:
                    norm_state = base_env.get_normalized_vessel_state()
                    _log_normalized_vessel_state(norm_state)
                    
                except ValueError as norm_error:
                    logger.error(f"Normalization error: {norm_error}")
                    # Optional: break the loop or continue based on requirements
                    # break

                # Optional delay between steps
                time.sleep(0.5)

            except Exception as step_error:
                logger.error(f"Error during step {step + 1}: {step_error}")
                # Decide whether to continue or break based on error severity
                # break

    except ImportError:
        logger.error("Could not import KerbinOrbitalEnvironment. Check module path.")
    except Exception as e:
        logger.error(f"Unexpected error during KSP connection test: {e}")
    finally:
        logger.info("\nKSP Connection Test Completed")

def _log_raw_vessel_state(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Logs raw vessel telemetry data with formatted output.
    
    Args:
        raw_state (Dict[str, Any]): Raw vessel state dictionary
    
    Returns:
        Dict[str, Any]: The input raw state dictionary
    """
    logging.getLogger(__name__).info("\nRaw Vessel Telemetry:")
    logging.getLogger(__name__).info(f"Altitude: {raw_state['altitude']:.2f} m")
    logging.getLogger(__name__).info(f"Velocity: {raw_state['velocity']} m/s")
    logging.getLogger(__name__).info(f"Pitch: {raw_state['pitch']:.2f}°")
    logging.getLogger(__name__).info(f"Roll: {raw_state['roll']:.2f}°")
    logging.getLogger(__name__).info(f"Yaw: {raw_state['yaw']:.2f}°")
    
    logging.getLogger(__name__).info("\nEngine and Mass:")
    logging.getLogger(__name__).info(f"Current Thrust: {raw_state['thrust']:.2f} N")
    logging.getLogger(__name__).info(f"Available Thrust: {raw_state['avail_thrust']:.2f} N")
    logging.getLogger(__name__).info(f"Total Mass: {raw_state['mass']:.2f} kg")
    logging.getLogger(__name__).info(f"Dry Mass: {raw_state['dry_mass']:.2f} kg")
    
    logging.getLogger(__name__).info("\nOrbital Parameters:")
    logging.getLogger(__name__).info(f"Apoapsis: {raw_state['apoapsis']:.2f} m")
    logging.getLogger(__name__).info(f"Periapsis: {raw_state['periapsis']:.2f} m")
    logging.getLogger(__name__).info(f"Eccentricity: {raw_state['eccentricity']:.4f}")
    logging.getLogger(__name__).info(f"Inclination: {raw_state['inclination']:.2f}°")
    
    logging.getLogger(__name__).info("\nResource Levels:")
    logging.getLogger(__name__).info(f"Fuel: {raw_state['fuel']:.2f} units")
    logging.getLogger(__name__).info(f"Oxidizer: {raw_state['oxidizer']:.2f} units")
    logging.getLogger(__name__).info(f"Electric Charge: {raw_state['electric_charge']:.2f} units")
    
    logging.getLogger(__name__).info("\nControl Settings:")
    logging.getLogger(__name__).info(f"Throttle: {raw_state['throttle']:.2f}")
    
    return raw_state

def _log_normalized_vessel_state(norm_state: Dict[str, Any]) -> None:
    """
    Logs normalized vessel state data with formatted output.
    
    Args:
        norm_state (Dict[str, Any]): Normalized vessel state dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("\nNormalized Vessel State:")
    logger.info(f"Normalized Altitude: {norm_state['altitude']:.4f}")
    logger.info(f"Normalized Velocity: {[f'{v:.4f}' for v in norm_state['velocity']]}")
    logger.info(f"Normalized Pitch: {norm_state['pitch']:.4f}")
    logger.info(f"Normalized Yaw: {norm_state['yaw']:.4f}")
    logger.info(f"Normalized Roll: {norm_state['roll']:.4f}")
    logger.info(f"Normalized Apoapsis: {norm_state['apoapsis']:.4f}")
    logger.info(f"Normalized Periapsis: {norm_state['periapsis']:.4f}")

# Allow direct script execution
if __name__ == "__main__":
    test_ksp_connection()