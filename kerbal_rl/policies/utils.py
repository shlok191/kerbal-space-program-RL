import jax.numpy as jnp
from typing import Dict, Any, Tuple

from kerbal_rl.policies.networks import (
    ActorCriticTrainState,
    initialize_networks,
)

from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

#######################################################
#
#  1. Defining functions for JAX - Python conversions
#     useful for training and debugging
#
#######################################################


def dict_to_array(state_dict: Dict[str, Any]) -> jnp.ndarray:
    """Converts the state dictionary to a valid JAX array!

    Args:
        state_dict (Dict): The state dictionary representing the environment state

    Returns:
        JAX array: A JAX array that can be passed into the Actor-Critic networks!
    """

    # Extracting the velocity components since it is a 3D vector
    velocity = state_dict.pop("velocity")

    # Convert remaining values to array and concatenate with velocity
    return jnp.concatenate([jnp.array(list(state_dict.values())), jnp.array(velocity)])


################################################
#
# 2. Defining Checkpointing related functions
#
################################################


def restore_train_state(
    directory: str,
    action_dim: int,
    observation_dim: int,
    actor_dense_dim: int,
    critic_dense_dim: int,
    actor_lr: float,
    critic_lr: float,
    max_to_keep: int,
) -> Tuple[ActorCriticTrainState, CheckpointManager, int]:
    """Returns training state from a checkpoint or initializes new instances

    Args:
        directory (str): The absolute path of the checkpoint directory
        action_dim (int): The action dimension of the Actor Network
        observation_dim (int): The observation dimension of the environment
        actor_dense_dim (int): The out dimensions of the Dense layers in the Actor Network
        critic_dense_dim (int): The out dimensions of the Dense layers in the Critic Network
        actor_lr (float): Learning Rate for the Actor Network
        critic_lr (float): Learning Rate for the Critic Network

    Returns:
        Tuple[ActorNetwork, CriticNetwork, ActorCriticTrainState, CheckpointManager, int]:

            - ActorNetwork: The created Cctor Network
            - CriticNetwork: The created Critic Network
            - ActorCriticTrainState: The loaded / created Training State
            - CheckpointManager: The checkpoint manager used for training
            - Int: The current global step count
    """

    # First we create our checkpoint manager!
    checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    
    checkpoint_manager = CheckpointManager(
        directory=directory,
        checkpointers={"train_state": checkpointer},
        options=options,
    )

    # Checking for existing checkpoints
    latest_step = checkpoint_manager.latest_step()
    
    if latest_step is not None:
    
        print(f"Restoring training from the step {latest_step}...")
        print(f"Best of luck for training! :)")
        
        restored_dict = checkpoint_manager.restore(latest_step)
        
        # Fetching the train state object
        state_dict = restored_dict["train_state"]
        train_state = ActorCriticTrainState(**state_dict)
        
        # Fetching the global steps
        start_step = train_state.step
        
        # Using the init. network function to get the networks (a bit lazy, I know!)
        actor_network, critic_network, _ = initialize_networks(
            action_dim=action_dim,
            observation_dim=observation_dim,
            actor_dense_dim=actor_dense_dim,
            critic_dense_dim=critic_dense_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )
        
    else:
        print("No checkpoint found; initializing a fresh TrainState.")
        print(f"Best of luck for training! :)")
        
        # Initializing a new training state if none is provided
        actor_network, critic_network, train_state = initialize_networks(
            action_dim=action_dim,
            observation_dim=observation_dim,
            actor_dense_dim=actor_dense_dim,
            critic_dense_dim=critic_dense_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )
        
        start_step = 0
    
    return actor_network, critic_network, train_state, checkpoint_manager, start_step
