# Importing all the needed libraries
from typing import Any

from kerbal_rl.environments import KerbinOrbitalEnvironment, MissionStatus
from kerbal_rl.policies.networks import (
    ActorNetwork,
    CriticNetwork,
    ActorCriticTrainState,
    GAE_function,
)

from kerbal_rl.policies.utils import dict_to_array, restore_train_state

import jax
import flax
import optax
import distrax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import orbax.checkpoint as ocp
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

import wandb

# Defining important hyperparameters
TOTAL_STEPS = 500_000
STEPS_PER_BATCH = 128
MINI_BATCH_SIZE = 32
TOTAL_EPOCHS = 16

EVAL_INTERVAL = 100

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.4

ACTOR_LR = 5e-4
CRITIC_LR = 8e-4

VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

CHECKPOINT_ITER = 512


def update_networks(
    training_state: ActorCriticTrainState,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    num_epochs: int,
    mini_batch_size: int,
    debug_print: bool = True,
) -> ActorCriticTrainState:
    """
    Implements the PPO network update algorithm!

    Here are the steps followed by me, inspired from the PPO paper:

        1. Compute logarithmic probabilities
        2. Creating mini-batches that we iterate over `epoch` times to update the weights
        4. Clipping policy ratios to prevent drastic policy changes --> creates a trust region, essentially!

    Args:
        training_state (TrainingState): Holds the model weights, optimizer states and other metadata
        observations jnp.ndarray: The collection of state observations across N steps
        actions jnp.ndarray: The collection of actions taken across N steps
        advantages (jnp.ndarray): The collection of GAE advantages across N steps
        rewards (jnp.ndarray): The collection of rewards across N steps
        dones (jnp.ndarray): The collection of done flags across N steps
        num_epochs (int): Number of times to iterate over the steps for policy updates
        mini_batch_size (int): Size of the mini-batches for policy updates per epoch
        debug_print (bool): Whether or not to do the debug print outs

    Returns:
        -   TrainingState: An updated training state with new network parameters! :)
        -   Dict[Any, Any]: A dictionary containing loss values and other metrics for W&B
    """

    def compute_ratio_func(
        actor_network,
        actor_params,
        actor_batch_stats,
        observations,
        actions,
        old_log_probs,
    ):
        """Computes the Ratio function R(theta) to compare policy changes!

        Args:
            actor_network (flax.linen.Module): The Actor Network
            actor_params (jnp.tree): A JAX PyTree denoting the parameters of the Actor Network
              actor_batch_stats (jnp.array): The batch statistics for the Actor Network
            observations (jnp.ndarray): The state observations for the given timesteps
            actions (jnp.ndarray): The actions taken by the new policy for the given timesteps
            old_log_probs (jnp.ndarray): The old logarithm of the probabilities of actions taken

        Returns:
            Tuple [Float, jnp.ndarray, jnp.ndarray]:
                - The ratio function R(theta)
                - The new probability distribution
                - The batch statistics for the actor network
        """

        # Removing the terminal state from the observations since no action needs to be taken here :)
        actor_observations = observations[:, :-1]

        actor_metadata = {"params": actor_params, "batch_stats": actor_batch_stats}

        # First, we compute the ratio function  which tells us the ratio of the
        # probabilities of taking certain actions under the current policy vs the old policy!
        (mean, std_dev), updates = actor_network.apply(
            actor_metadata,
            actor_observations,
            training=True,
            mutable=["batch_stats"],
        )

        # Breaking down the output into constituents :)
        new_prob_dist = distrax.Normal(loc=mean, scale=std_dev)

        new_log_probs = new_prob_dist.log_prob(actions).sum(axis=-1)

        # Calculating the ratio --> the logarithm and exponentials make it easier to compute!
        ratio = jnp.exp(new_log_probs - old_log_probs)

        return ratio, new_prob_dist, updates["batch_stats"]

    def compute_actor_loss(policy_ratio, gae_advantages):
        """Computes the loss for the Actor Network

        Args:
            policy_ratio (Float): The policy ratio discussed above
            gae_advantages (jnp.ndarray): The GAE advantages computed for the timesteps

        Returns:
            Float: The loss value for the Actor Network
        """

        # Clipping the ratio to avoid drastic changes!
        clipped_ratio = jnp.clip(policy_ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)

        # Taking the minimum of the surrogate objective
        surrogate_objective = jnp.minimum(
            policy_ratio * gae_advantages, clipped_ratio * gae_advantages
        )

        # Taking the average across all time steps :)
        actor_loss = -jnp.mean(surrogate_objective)

        return actor_loss

    def compute_critic_loss(
        critic_network, critic_params, critic_batch_stats, observations, returns
    ):
        """Computes the loss for the Critic Network

        Args:
            critic_network (flax.linen.Module): The Critic Network
            critic_params (jax.PyTree): The parameters associated with the Critic Network
            critic_batch_stats (jax.PyTree): The batch statistics for the Critic Network
            observations (jax.ndarray): A collection of state values across the timesteps
            returns (jax.ndarray): The obtained returns (fut. rewards + discounting) from each time step

        Returns:
            Tuple [Float, jnp.ndarray]:

                - Float: The loss value for the Critic Network
                - jnp.ndarray: The batch statistics for the Critic Network
        """

        critic_metadata = {"params": critic_params, "batch_stats": critic_batch_stats}

        # Fetching the predictions of the Critic Network
        preds, critic_batch_stats = critic_network.apply(
            critic_metadata,
            observations,
            training=True,
            mutable=["batch_stats"],
        )

        # We use a simple MSE loss for the Critic Network
        critic_loss = jnp.mean(jnp.square(preds - returns))

        return critic_loss, critic_batch_stats

    def compute_returns(rewards, values, dones, gamma):
        """
        Computes returns using Bellman backup.

        Args:
            rewards (jnp.ndarray): Rewards for each timestep, shape [batch_size, timesteps].
            values (jnp.ndarray): Value predictions from the Critic, shape [batch_size, timesteps + 1].
            dones (jnp.ndarray): Boolean array indicating episode termination, shape [batch_size, timesteps].
            gamma (float): Discount factor for future rewards.

        Returns:
            jnp.ndarray: Computed returns for each timestep, shape [batch_size, timesteps].
        """

        returns = []
        next_return = values[
            :, -1
        ]  # Start from the bootstrap value (final state's value)

        for t in reversed(range(rewards.shape[1])):

            next_return = rewards[:, t] + gamma * next_return * (1 - dones[:, t])
            returns.insert(0, next_return)

        return jnp.stack(returns, axis=1)

    def ppo_loss(
        actor_network,
        critic_network,
        actor_params,
        actor_batch_stats,
        critic_params,
        critic_batch_stats,
        observations,
        actions,
        advantages,
        old_log_probs,
        returns,
    ):
        """
        Computes the PPO loss function!

        Args:
            actor_network (flax.linen.Module): The Actor Network
            critic_network (flax.linen.Module): The Critic Network
            actor_params (jax.PyTree): The Actor Network parameters
            actor_batch_stats (jax.PyTree): The Actor Network batch statistics
            critic_params (Any): The Critic Network parameters
            critic_batch_stats (jax.PyTree): The Critic Network batch statistics
            observations (jnp.ndarray): The observations
            actions (jnp.ndarray): The actions taken
            advantages (jnp.ndarray): The GAE advantages
            old_log_probs (jnp.ndarray): The old log probabilities

        Returns: The loss value
        """

        # Fetching the ratio value and the current distribution of actions
        policy_ratio, current_dist, actor_batch_stats = compute_ratio_func(
            actor_network,
            actor_params,
            actor_batch_stats,
            observations,
            actions,
            old_log_probs,
        )

        # Calculating the respective losses
        actor_loss = compute_actor_loss(policy_ratio, advantages)
        critic_loss, critic_batch_stats = compute_critic_loss(
            critic_network, critic_params, critic_batch_stats, observations, returns
        )

        # Calculating the enrropy of the probability distribution
        entropy = jnp.mean(current_dist.entropy())

        # According to the PPO paper, this is what the loss should look like!
        total_loss = actor_loss + VALUE_LOSS_COEF * critic_loss - ENTROPY_COEF * entropy

        return (
            total_loss,
            actor_loss,
            critic_loss,
            actor_batch_stats,
            critic_batch_stats,
            policy_ratio,
            entropy,
        )

    def param_diff(old_params, new_params):
        """Compute the L2 norm of the difference between old and new parameters."""
        old_flat, _ = ravel_pytree(old_params)
        new_flat, _ = ravel_pytree(new_params)
        return jnp.linalg.norm(old_flat - new_flat)

    def grads_norm(grads):
        """Compute the L2 norm of the flattened gradients."""
        flat_grads, _ = ravel_pytree(grads)
        return jnp.linalg.norm(flat_grads)

    def update_step(training_state, batch):
        """
        Performs a single gradient update step.

        Uses JAX's automatic differentiation to compute gradients
        and update network parameters!

        Args:
            training_state (TrainingState): The current training state
            batch (Tuple): A tuple containing the mini-batch data

        Returns:
            TrainingState: The updated training state :)
        """

        observations, actions, advantages, old_log_probs, returns = batch

        old_actor_params = training_state.params
        old_critic_params = training_state.critic_params

        def loss_fn(training_state: ActorCriticTrainState):
            """The loss function for the PPO updates

            Args:
                training_state (ActorCriticTrainState): The training state containing the metadata

            Returns:
                Tuple [float, Tuple[float, float, jnp.array, jnp.array, float, float]]:

                    - float: The total loss calculated by the network
                    - float: The actor network loss
                    - float: The critic network loss
                    - jax.array: The updated actor batch statistics
                    - jax.array: The updated critic batch statistics
                    - ratio: The ratio of the new actor distribution vs the old one :)
                    - entropy: The calculated entropy of the model
            """

            # Sorry, the black formatter works this out a bit weirdly!
            (
                total_loss,
                actor_loss,
                critic_loss,
                actor_batch_stats,
                critic_batch_stats,
                ratio,
                entropy,
            ) = ppo_loss(
                actor_network=actor_network,
                critic_network=critic_network,
                actor_params=training_state.params,
                critic_params=training_state.critic_params,
                actor_batch_stats=actor_batch_stats,
                critic_batch_stats=critic_batch_stats,
                observations=observations,
                actions=actions,
                advantages=advantages,
                old_log_probs=old_log_probs,
                returns=returns,
            )

            return total_loss, (
                actor_loss,
                critic_loss,
                actor_batch_stats,
                critic_batch_stats,
                ratio,
                entropy,
            )

        # Computing the gradients with respect to the actor loss and the critic loss
        # This then returns 2 gradients! :)
        (
            total_loss,
            (
                actor_loss,
                critic_loss,
                actor_batch_stats,
                critic_batch_stats,
                ratio,
                entropy,
            ),
        ), grads = jax.value_and_grad(loss_fn, argnums=(0, 2), has_aux=True)(
            train_state
        )

        actor_grads, critic_grads = grads

        # Applying the gradients! This also increments the global step count :)
        training_state = training_state.apply_gradients(
            actor_grads=actor_grads,
            critic_grads=critic_grads,
            actor_batch_stats=actor_batch_stats,
            critic_batch_stats=critic_batch_stats,
        )

        # Creating the W&B metrics
        log_dict = {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "rewards/mean": jnp.mean(rewards),
            "rewards/std": jnp.std(rewards),
            "rewards/min": jnp.min(rewards),
            "rewards/max": jnp.max(rewards),
            "policy_ratio_mean": jnp.mean(ratio),
            "advantage_mean": jnp.mean(advantages),
            "entropy": entropy,
        }

        # Printing the model updates!
        if debug_print:

            actor_grad_norm = grads_norm(grads[0])
            critic_grad_norm = grads_norm(grads[1])

            actor_param_diff = param_diff(old_actor_params, training_state.params)
            critic_param_diff = param_diff(
                old_critic_params, training_state.critic_params
            )

            print(f"[DEBUG] Actor Grad Norm: {actor_grad_norm:.6f}")
            print(f"Critic Grad Norm: {critic_grad_norm:.6f}")

            print(f"[DEBUG] Actor Param L2 Δ: {actor_param_diff:.6f}")
            print(f"Critic Param L2 Δ: {critic_param_diff:.6f}")

        return training_state, log_dict

    def prepare_batches(observations, actions, advantages, old_log_probs, returns):
        """
        Shuffles and splits the data into mini-batches for PPO updates!

        Args:
            observations (List[Dict[str, jnp.ndarray]): List of observations
            actions (List[jnp.ndarray]): List of actions
            advantages (jnp.ndarray): List of GAE advantages
            old_log_probs (jnp.ndarray): List of old log probabilities
            returns (jnp.ndarray): List of the returns per timestep

        Yields:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

                - Mini-batches of observations, actions, advantages, old log probabilities, and returns!
        """

        dataset_size = len(observations)

        # Getting random indices separating the data into mini-batches!
        indices = jax.random.permutation(jax.random.PRNGKey(42), dataset_size)

        # Yielding the mini-batches as the funciton is called!
        for start in range(0, dataset_size, mini_batch_size):

            batch_indices = indices[start : start + mini_batch_size]

            yield (
                jnp.array([observations[i] for i in batch_indices]),
                jnp.array([actions[i] for i in batch_indices]),
                jnp.array([advantages[i] for i in batch_indices]),
                jnp.array([old_log_probs[i] for i in batch_indices]),
                jnp.array([returns[i] for i in batch_indices]),
            )

    actor_observations = observations[:, :-1]

    # Main training loop
    for _ in range(num_epochs):

        actor_metadata = {
            "params": training_state.params,
            "batch_stats": training_state.actor_batch_stats,
        }

        # Fetching the current mean and standard deviation values before updating the actor network!
        mean, std_dev = actor_network.apply(
            actor_metadata,
            actor_observations,
            training=False,
        )

        # Converting the mean and standard dev. to logarithmic probabilities :)
        old_prob_dist = distrax.Normal(loc=mean, scale=std_dev)
        old_log_probs = old_prob_dist.log_prob(actions).sum(axis=-1)

        critic_metadata = {
            "params": training_state.critic_params,
            "batch_stats": training_state.critic_batch_stats,
        }

        # Fetching the values from the Critic Network
        values, critic_batch_stats = critic_network.apply(
            critic_metadata, observations, training=True, mutable=["batch_stats"]
        )

        # Updating the training state!
        training_state = training_state.replace(critic_batch_stats=critic_batch_stats)

        # Removing the final dimension since it only holds 1 value
        values = values.squeeze(-1)

        # Calculating the GAE advantages and the returns for all time steps :)
        advantages = GAE_function(rewards, values, dones, GAMMA, LAMBDA)
        returns = compute_returns(rewards, values, dones, GAMMA)

        batch_metrics = []

        # Finally, we update the network parameters using mini batches!
        for batch in prepare_batches(
            observations, actions, advantages, old_log_probs, returns
        ):

            training_state, batch_metric = update_step(training_state, batch)
            batch_metrics.append(batch_metric)

        # Taking the average across all batches! :)
        mean_metrics = {}

        for key in batch_metrics[0].keys():

            values = [metrics[key] for metrics in batch_metrics]
            mean_metrics[key] = jnp.mean(jnp.array(values))

        # Adding the step count!
        mean_metrics["steps"] = training_state.steps

        # Logging the averaged metrics
        wandb.log(mean_metrics)

    return training_state


def collect_experience(
    env: KerbinOrbitalEnvironment,
    actor: flax.linen.Module,
    actor_params: Any,
    actor_batch_stats: Any,
    steps: int,
    key: jax.random.PRNGKey,
    debug_print: bool = False,
):
    """Collects steps from the environment using the actor network.

    Args:
        env (KerbinOrbitalEnvironment): A Kerbin Orbital Network
        actor (flax.linen.Module): The Actor network taking actions
        actor_params (Any): The Actor Network weights
        actor_batch_stats (Any): The batch statistics for the Actor Network
        steps (int): The number of steps to take!
        key (jax.random.PRNGKey): The randomized key used for sampling actions
        debug_print (bool): Whether or not to do debugging print outs
    """

    if debug_print:
        print("===== Starting a new observation collection sequence! =====\n")
        print(f"We will be collecting {steps} steps")

        print("\n--- Episode Statistics ---")
        print("Episode Length | Total Reward | Terminal State")
        print("-" * 50 + "\n")

    # Keeps track of the observations, actions, rewards and dones!
    observations = []
    actions = []
    rewards = []
    dones = []

    episode_reward = 0
    action_keys = jax.random.split(key, steps)

    for step in range(steps):

        # Fetches the current normalized vessel state
        curr_vehicle_state = dict_to_array(env.get_normalized_vessel_state())

        actor_metadata = {"params": actor_params, "batch_stats": actor_batch_stats}
        mean, std_dev = actor.apply(actor_metadata, curr_vehicle_state, training=False)

        # Sampling an action from the Gaussian distribution :)
        action_distribution = distrax.Normal(loc=mean, scale=std_dev)
        sampled_action = action_distribution.sample(seed=action_keys[step])

        # Converting the action to a dictionary
        action = {
            "throttle": sampled_action[0],
            "pitch": sampled_action[1],
            "yaw": sampled_action[2],
            "roll": sampled_action[3],
        }

        # Taking the action in the environment
        reward, done = env.step(action)

        # Appending the current state, the action taken, the reward and IF we reach a terminal state!
        observations.append(curr_vehicle_state)
        actions.append(sampled_action)
        rewards.append(reward)
        dones.append(done is not MissionStatus.IN_PROGRESS)

        episode_reward += reward

        # Restarting the episode if we reach a terminal state!
        if done is not MissionStatus.IN_PROGRESS and debug_print:

            print(f"{step} | {episode_reward} | {done}")

            episode_reward = 0
            env.restart_episode()

    # This means we ended at a non-terminal state
    if dones[-1] is False:

        print(f"{steps} | {episode_reward} | {done}")
        env.restart_episode()

    # Adding the final state to the observations
    final_vehicle_state = dict_to_array(env.get_normalized_vessel_state())
    observations.append(final_vehicle_state)

    # Converting all values to JAX arrays
    observations = jnp.expand_dims(jnp.array(observations), axis=0)
    actions = jnp.expand_dims(jnp.array(actions), axis=0)
    rewards = jnp.expand_dims(jnp.array(rewards), axis=0)
    dones = jnp.expand_dims(jnp.array(dones), axis=0)

    # Normalizing the rewards to stabilize training
    rewards_mean = float(jnp.mean(rewards, axis=1, keepdims=True).item())
    rewards_std = float(jnp.std(rewards, axis=1, keepdims=True).item()) + 1e-8

    normalized_rewards = (rewards - rewards_mean) / rewards_std

    if debug_print:

        print("\n=== Reward Statistics ===")

        print(f"Raw Rewards - Mean: {rewards_mean:.3f}, Std: {rewards_std:.3f}")
        print(f"Normalized Rewards - Mean: {jnp.mean(normalized_rewards).item():.3f}")
        print(f"Normalized Rewards - Std Dev: {jnp.std(normalized_rewards).item():.3f}")

        print(f"{'=' * 50} \n")

    # Returning the associated values
    return observations, actions, normalized_rewards, dones


def train_ppo(
    environment: KerbinOrbitalEnvironment,
    actor_network: ActorNetwork,
    critic_network: CriticNetwork,
    training_state: ActorCriticTrainState,
    checkpoint_manager: CheckpointManager,
    total_steps: int,
    steps_per_batch: int,
    num_epochs: int,
    mini_batch_size: int,
    start_step: int,
):
    """Defines the training loop code for PPO training :)

    Args:
        environment (KerbinOrbitalEnvironment): The environment to train the agent in
        actor_network (ActorNetwork): The Actor network that defines the actions to take
        critic_network (CriticNetwork): The Critic network which returns value functions
        training_state (ActorCriticTrainState): The training state containing important metadata
        checkpoint_manager (CheckpointManager): The checkpoint manager used to save checkpoints
        total_steps (int): The total amount of steps to process
        steps_per_batch (int): The total amount of steps to process per batch
        num_epochs (int): The number of times each batch is iterated through
        mini_batch_size (int): The mini batch the larger -- 1024 batch -- is broken down into
        start_step (int): The global starting step if we are resuming from a checkpoint
    """

    master_key = jax.random.PRNGKey(0)

    # In each iteration, we traverse a set amount of steps which will give us values to improve the network!
    
    start_iter = start_step // steps_per_batch
    end_iter =  total_steps // steps_per_batch
    
    for iteration in range(start_iter, end_iter):

        # Splitting the keys into new sets
        master_key, collect_key = jax.random.split(master_key)

        # Collecting observations for the pre-decided number of step values
        observations, actions, normalized_rewards, dones = collect_experience(
            env=environment,
            actor=actor_network,
            actor_params=training_state.params,
            actor_batch_stats=training_state.actor_batch_stats,
            steps=steps_per_batch,
            key=collect_key,
        )

        # Fetching future value predictions from the Critic Network
        critic_metadata = {
            "params": training_state.critic_params,
            "batch_stats": training_state.critic_batch_stats,
        }

        values, critic_batch_stats = critic_network.apply(
            critic_metadata, observations, training=True, mutable=["batch_stats"]
        )

        values = values.squeeze(-1)

        training_state = training_state.replace(critic_batch_stats=critic_batch_stats)

        # Calculating the GAE advantages
        advantages = GAE_function(normalized_rewards, values, dones)

        # Update networks using PPO objectives
        training_state = update_networks(
            training_state,
            observations,
            actions,
            advantages,
            normalized_rewards,
            dones,
            num_epochs,
            mini_batch_size,
        )

        # Saving the checkpoint every X iterations
        if iteration % CHECKPOINT_ITER == 0:

            state_dict = {
                "actor_params": training_state.params,
                "critic_params": training_state.critic_params,
                "actor_batch_stats": training_state.actor_batch_stats,
                "critic_batch_stats": training_state.critic_batch_stats,
                "actor_opt_state": training_state.opt_state,
                "critic_opt_state": training_state.critic_opt_state,
                "steps": training_state.steps,
            }

            checkpoint_manager.save(iteration, {"train_state": state_dict})

            print(f"Saved checkpoint at step {iteration}!")


if __name__ == "__main__":

    # Initializing the environment for the orbital mission!
    environment = KerbinOrbitalEnvironment(
        ip_address="127.0.0.1",
        rpc_port=50000,
        stream_port=50001,
        connection_name="Orbital Mission #1",
        vessel_name="Orbital Vessel #1",
        max_steps=1000,
        target_apoapsis=75000,
        target_periapsis=75000,
        max_altitude=85000,
        rcs_enabled=False,
        step_sim_time=0.5,
    )

    # Initializing the wandb environment for logging purposes
    wandb.init(
        project="ksp-orbital-insertion",
        config={
            "total_steps": TOTAL_STEPS,
            "steps_per_batch": STEPS_PER_BATCH,
            "mini_batch_size": MINI_BATCH_SIZE,
            "epochs": TOTAL_EPOCHS,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "clip_epsilon": CLIP_EPSILON,
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "value_loss_coef": VALUE_LOSS_COEF,
            "entropy_coef": ENTROPY_COEF,
            "target_apoapsis": environment.target_apoapsis,
            "target_periapsis": environment.target_periapsis,
        },
    )

    # Fetching variables to fetch model hyperparameters
    action_dim = len(environment.action_space)
    observation_dim = len(environment.get_normalized_vessel_state())

    # Fetching the models from an existing checkpoint or making a new one!
    actor_network, critic_network, train_state, checkpoint_manager, step = (
        restore_train_state(
            directory="/Users/sabarwal/work/projects/kerbal-space-program-RL/checkpoints/orbital-checkpoints",
            action_dim=action_dim,
            observation_dim=observation_dim,
            actor_dense_dim=256,
            critic_dense_dim=256,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            max_to_keep=5,
        )
    )

    # Training the PPO model now!
    train_ppo(
        environment,
        actor_network,
        critic_network,
        train_state,
        checkpoint_manager,
        TOTAL_STEPS,
        STEPS_PER_BATCH,
        TOTAL_EPOCHS,
        MINI_BATCH_SIZE,
        step,
    )
