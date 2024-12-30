from kerbal_rl.environments import KerbinOrbitalEnvironment, MissionStatus
from kerbal_rl.policies.utils import ActorNetwork, CriticNetwork, GAE_function

import jax.numpy as jnp
import optax
import flax.training.train_state as TrainState
from typing import Any
import flax
from typing import List, Dict
import time

# Defining hyperparameters
NUM_ITERATIONS = 20000
EVAL_INTERVAL = 100
BATCH_SIZE = 2048
MINIBATCH_SIZE = 64
EPOCHS = 4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
VALUE_LOSS_COEF = 0.5 
ENTROPY_COEF = 0.01

@flax.struct.dataclass  
class TrainingState:
    actor_params: Any      # Weights of the actor network
    critic_params: Any     # Weights of the critic network
    actor_opt_state: Any   # The optimizer state values
    critic_opt_state: Any  # The optimizer state values
    steps: int             # The total number of steps taken thus far

import jax
import jax.numpy as jnp
import flax
import optax

def update_networks(
    training_state: TrainingState,
    observations: List[Dict[str, jnp.ndarray]],
    actions: List[jnp.ndarray],
    advantages: jnp.ndarray,
    num_epochs: int,
    mini_batch_size: int,
    actor_optimizer,
    critic_optimizer
) -> TrainingState:
    """
    Implements the PPO network update algorithm.
    
    The Core PPO Update Strategy:
    
        1. Compute old log probabilities
        2. Creating mini-batches that are iterated over `epoch` times
        4. Clipping policy ratios to prevent drastic policy changes --> makes PPO mroe stable! :)
    
    Args:
        training_state (TrainingState): The Current training state since JAX is functional
        observations (List[Dict]): The collected dictionary of observation across steps
        actions (List[jnp.ndarray]): The collected actions taken across steps
        advantages (jnp.ndarray): The collected GAE advantages across steps
        num_epochs (int): Number of times to iterate over the dataset
        mini_batch_size (int): Size of mini-batches for policy updates per epoch
    
    Returns:
        TrainingState: Updated training state with new network parameters
    """
    
    def compute_log_probabilities(actor_params, observations, actions):
        """
        Computes log probabilities of actions under the current policy.
        
        Args:
            actor_params (Any): The Actor Network parameters -- again, since JAX is functional!
            observations (List[Dict[str, jnp.ndarray]]): List of observations
            actions (List[jnp.ndarray]): List of actions
        
        Returns:
            jnp.ndarray: Logarithmic probabilities of the actions
        """
        
        action_dist = actor_network.apply(actor_params, observations)
        return action_dist.log_prob(actions)
    
    # Compute old log probabilities (for importance sampling)
    old_log_probs = compute_log_probabilities(
        training_state.actor_params, 
        observations, 
        actions
    )
    

    def prepare_batches(observations, actions, advantages, old_log_probs):
        """
        Shuffles and splits the data into mini-batches for PPO updates!
        
        Args:
            observations (List[Dict[str, jnp.ndarray]): List of observations
            actions (List[jnp.ndarray]): List of actions
            advantages (jnp.ndarray): List of GAE advantages
            old_log_probs (jnp.ndarray): List of old log probabilities
        """
        
        
        dataset_size = len(observations)
        
        # Getting random indices separating the data into mini-batches
        indices = jax.random.permutation(
            jax.random.PRNGKey(int(time.time())), 
            dataset_size
        )
        
        # Yielding the mini-batches as the funciton is called!
        for start in range(0, dataset_size, mini_batch_size):
            
            batch_indices = indices[start:start + mini_batch_size]
            
            yield (
                jnp.array([observations[i] for i in batch_indices]),
                jnp.array([actions[i] for i in batch_indices]),
                jnp.array([advantages[i] for i in batch_indices]),
                jnp.array([old_log_probs[i] for i in batch_indices])
            )
    
    # PPO loss computation
    def ppo_loss(actor_params, critic_params, 
            observations, actions, advantages, old_log_probs):
        """
        Computes the PPO loss function!
        
        Args:
            actor_params (Any): The Actor Network parameters
            critic_params (Any): The Critic Network parameters
            observations (jnp.ndarray): The observations
            actions (jnp.ndarray): The actions taken
            advantages (jnp.ndarray): The GAE advantages
            old_log_probs (jnp.ndarray): The old log probabilities
            
        Returns: The loss value
        """
        
        # Computing current log probabilities
        current_log_probs = compute_log_probabilities(
            actor_params, observations, actions
        )
        
        # Computing the policy ratio
        actor_ratio = jnp.exp(current_log_probs - old_log_probs)
        
        # Clipping surrogate objective
        clipped_ratio = jnp.clip(
            actor_ratio, 
            1 - CLIP_EPSILON, 
            1 + CLIP_EPSILON
        )
        
        # Getting the loss for the actor network
        actor_loss = -jnp.mean(
            jnp.min(
                actor_ratio * advantages,
                clipped_ratio * advantages
            )
        )
        
        # Critic or Value function loss (MSE)
        critic_pred = critic_network.apply(critic_params, observations)
        critic_loss = jnp.mean(jnp.square(critic_pred - advantages))
        
        # Defining an entropy bonus to encourage exploration
        entropy = -jnp.mean(
            jax.scipy.special.entr(
                actor_network.apply(actor_params, observations).probs
            )
        )
        
        # Composite loss
        total_loss = (
            actor_loss + 
            VALUE_LOSS_COEF * critic_loss - 
            ENTROPY_COEF * entropy
        )
        
        return total_loss
    
    # Gradient update function
    @jax.jit
    def update_step(training_state, batch):
        """
        Performs a single gradient update step.
        
        Uses JAX's automatic differentiation to compute gradients
        and update network parameters.
        """
        
        observations, actions, advantages, old_log_probs = batch
        
        # Compute gradients
        loss_grad_fn = jax.value_and_grad(ppo_loss, argnums=(0, 1))
        
        loss, (actor_grads, critic_grads) = loss_grad_fn(
            training_state.actor_params,
            training_state.critic_params,
            observations,
            actions,
            advantages,
            old_log_probs
        )
        
        # Updating the actor network
        actor_updates, new_actor_opt_state = actor_optimizer.update(
            actor_grads, 
            training_state.actor_opt_state
        )
        
        new_actor_params = optax.apply_updates(
            training_state.actor_params, 
            actor_updates
        )
        
        # Updating the critic network
        critic_updates, new_critic_opt_state = critic_optimizer.update(
            critic_grads, 
            training_state.critic_opt_state
        )
        
        new_critic_params = optax.apply_updates(
            training_state.critic_params, 
            critic_updates
        )
        
        # Updating training state
        return TrainingState(
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            actor_opt_state=new_actor_opt_state,
            critic_opt_state=new_critic_opt_state,
            steps=training_state.steps + 1
        )
    
    # Main training loop
    for _ in range(num_epochs):
        
        for batch in prepare_batches(observations, actions, advantages, old_log_probs):
            training_state = update_step(training_state, batch)
    
    return training_state

def collect_experience(
    env: KerbinOrbitalEnvironment,
    actor: flax.linen.Module,
    actor_params: Any,
    steps: int
):
    """Collects steps from the environment using the actor network.

    Args:
        env (KerbinOrbitalEnvironment): A Kerbin Orbital Network
        actor (flax.linen.Module): The Actor network taking actions
        steps (int): The number of steps to take!
    """
    
    # Keeps track of the observations, actions, rewards and dones!
    observations = []
    actions = []
    rewards = []
    dones = []
    
    for _ in range(steps):
        
        # Fetches the current normalized vessel state
        curr_vehicle_state = dict_to_array(env.get_normalized_vessel_state())
        action = actor.apply(actor_params, curr_vehicle_state, training=False)
        
        # Converting the action to a dictionary!
        action = {
            "throttle": action[0],
            "pitch": action[1],
            "yaw": action[2],
            "roll": action[3]
        }
        
        # Taking the action in the environment
        reward, done = env.step(action)
        
        # Appending the current state, the action taken, the reward and IF we reach a terminal state!
        observations.append(curr_vehicle_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done is not None)
        
        if done == MissionStatus.COMPLETED:
            break
    
    return observations, actions, rewards, dones

def dict_to_array(state_dict: Dict[str, Any]) -> jnp.ndarray:
    """Converts the state dictionary to a valid JAX array!

    Args:
        state_dict (Dict): The state dictionary representing the environment state

    Returns:
        JAX array: A JAX array that can be passed into the Actor-Critic networks!
    """
    
    # Extracting the velocity components since it is a 3D vector
    velocity = state_dict.pop('velocity')
    
    # Convert remaining values to array and concatenate with velocity
    return jnp.concatenate([jnp.array(list(state_dict.values())), jnp.array(velocity)])

def train_ppo(
    environment: KerbinOrbitalEnvironment,
    actor_network: ActorNetwork,
    critic_network: CriticNetwork,
    total_steps: int = 500_000,
    steps_per_batch: int = 1024,
    num_epochs = 10,
    mini_batch_size = 32
):
    """Defines the training loop code for PPO training :)

    Args:
        environment (KerbinOrbitalEnvironment): The environment to train the agent in
        actor_network (ActorNetwork): The Actor network that defines the actions to take
        critic_network (CriticNetwork): The Critic network which returns value functions
        total_steps (int, optional): The total amount of steps to process. Defaults to 1_000_000.
        steps_per_batch (int, optional): The total amount of steps to process per batch. Defaults to 1024.
        num_epochs (int, optional): The number of times each batch is iterated through. Defaults to 10.
        mini_batch_size (int, optional): The mini batch the larger -- 1024 batch -- is broken down into. Defaults to 32.
    """

    
    # Initialize the network parameters
    key = jax.random.PRNGKey(0)
    key, actor_key, critic_key = jax.random.split(key, 3)
    
    # Fetching the current observation for initialization
    curr_observation = dict_to_array(environment.get_normalized_vessel_state())
    X = jnp.expand_dims(curr_observation, axis=0)
    
    # Initialize the network parameters
    actor_params = actor_network.init(actor_key, curr_observation, training=True)
    critic_params = critic_network.init(critic_key, curr_observation, training=True)
    
    # Initializing the optimizers
    actor_optimizer = optax.adam(ACTOR_LR)
    critic_optimizer = optax.adam(CRITIC_LR)
    
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    training_state = TrainingState(
        actor_params=actor_params,
        critic_params=critic_params,
        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
        steps=0
    )
    
    for _ in range(total_steps // steps_per_batch):
    
        observations, actions, rewards, dones = collect_experience(
            env=environment,
            actor=actor_network,
            actor_params=actor_params,
            steps=steps_per_batch
        )
        
        values = critic_network.apply(training_state.critic_params, curr_observation, observations)
        advantages = GAE_function(rewards, values, dones)
        
        # Update networks using PPO objectives
        training_state = update_networks(
            training_state, 
            observations, 
            actions, 
            advantages, 
            num_epochs, 
            mini_batch_size,
            actor_optimizer,
            critic_optimizer
        )

        print("Performed one iteration!")
            
if __name__ == "__main__":
    
    # Initializing the environment for the orbital mission!
    environment = KerbinOrbitalEnvironment(
        ip_address="127.0.0.1",
        rpc_port=50000,
        stream_port=50001,
        connection_name="Orbital Mission #1",
        vessel_name="Orbital Vessel #1",
        max_steps=10000,
        target_apoapsis=75000,
        target_periapsis=75000,
        max_altitude=80000,
        rcs_enabled=False
    )
    
    # Initializing the Actor and Critic Networks
    actor_network = ActorNetwork(action_dim=len(environment.action_space))
    critic_network = CriticNetwork()
    
    # Training the PPO model
    train_ppo(environment, actor_network, critic_network)