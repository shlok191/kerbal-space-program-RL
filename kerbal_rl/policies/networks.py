from typing import Tuple, Any, Callable

import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax
import jax

##############################################
#
# 1. Defining the Actor and Critic Networks
#
##############################################


class ActorNetwork(nn.Module):
    """Generates the Actor Network responsible for taking actions

    Args:
        action_dim (int): The number of values to predict for the action space
        dense_dim (int): The output dimensions for the Dense layers
    """

    # Defining the model parameters
    action_dim: int
    dense_dim: int

    def setup(self):
        """Initialize the layers of the Actor Network.

        Includes two hidden layers with batch normalization, followed by:

            - mean_layer: Predicts the mean for each action dimension.
            - log_std_layer: Predicts the log standard deviation.

        We do this because the return values should fit a continuous spectrum :)
        """

        # One point to note --> dense_dim is the output dimension!
        # The input dimension is inferred during the first forward pass by Flax.

        # I found this to be a very cool mechanism :)
        self.dense1 = nn.Dense(
            self.dense_dim, kernel_init=nn.initializers.xavier_uniform()
        )

        self.dense2 = nn.Dense(
            self.dense_dim, kernel_init=nn.initializers.xavier_uniform()
        )

        self.batch_norm_1 = nn.BatchNorm()
        self.batch_norm_2 = nn.BatchNorm()

        # Gives the mean and variance for each action in the continuous action space
        self.mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.01))
        self.log_std_dev = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.normal(0.01)
        )

    def __call__(self, x, training: bool = True) -> jnp.ndarray:
        """Performs the forward pass given an input JAX ndarray X.

        Args:
            x (jnp.ndarray): A JAX-Numpy ndarray denoting the input [batch_size, observation_dim]
            training (bool): A bool value indicating whether the model is in training mode. Effects batch norm!

        Returns:
            Tuple [jnp.ndarray, jnp.ndarray]

                - Mean Array: An output tensor of shape [batch_size, action_dim]
                - Std. Deviation Array: An output tensor of shape [batch_size, action_dim]
        """

        x = nn.relu(self.dense1(x))
        x = self.batch_norm_1(x, use_running_average=not training)

        x = nn.relu(self.dense2(x))
        x = self.batch_norm_2(x, use_running_average=not training)

        mean = self.mean(x)
        log_std = self.log_std_dev(x)

        # Ensuring the value is always positive!
        std = jnp.exp(log_std)

        return mean, std


class CriticNetwork(nn.Module):
    """Generates the Critic Network responsible for estimating the Value Function

    Args:
        dense_dim (int): The output dimensions for the Dense layers
    """

    # Defining the model parameters
    dense_dim: int

    def setup(self):
        """Defines the value network which estimates
        the estimated values of given states.
        """

        self.dense1 = nn.Dense(
            self.dense_dim, kernel_init=nn.initializers.xavier_uniform()
        )
        self.dense2 = nn.Dense(
            self.dense_dim, kernel_init=nn.initializers.xavier_uniform()
        )

        self.batch_norm_1 = nn.BatchNorm()
        self.batch_norm_2 = nn.BatchNorm()

        # Our value function only outputs one value denoting the estimated V(s)!
        self.value = nn.Dense(1, kernel_init=nn.initializers.normal(0.1))

    def __call__(self, x, training: bool = True):
        """Returns the expected value of the given state.

        Args:
            x (jnp.ndarray): A JAX-Numpy ndarray denoting the input [batch_size, observation_dim]
            training (bool): A bool value indicating whether the model is in training mode. Effects batch norm!

        Returns:
            jnp.nadarray: Output JAX-Numpy ndarray of shape [batch_size, 1]
        """

        x = nn.relu(self.dense1(x))
        x = self.batch_norm_1(x, use_running_average=not training)

        x = nn.relu(self.dense2(x))
        x = self.batch_norm_2(x, use_running_average=not training)

        return self.value(x)


#########################################################################
#
# 2. Defining the Training State and Network Initialization Functions
#
#########################################################################


# Necessary due to the JAX functional approach, I am not complaining! :)
@flax.struct.dataclass
class ActorCriticTrainState(train_state.TrainState):

    # Defining any new parameters needed for the Actor Network
    actor_batch_stats: Any

    # Defining the needed parameters for the Critic Network
    critic_params: Any
    critic_batch_stats: Any
    critic_opt_state: Any
    critic_apply_fn: Callable
    critic_tx: optax.GradientTransformation

    # Defining a custom gradient function to update both networks
    def apply_gradients(
        self,
        *,
        actor_grads,
        critic_grads,
        actor_batch_stats=None,
        critic_batch_stats=None
    ) -> "ActorCriticTrainState":

        # Updating the Actor parameters
        
        actor_updates, actor_opt_state = self.tx.update(actor_grads, self.opt_state, self.params)
        actor_params = optax.apply_updates(self.params, actor_updates)

        # Updating the Critic parameters
        critic_updates, critic_opt_state = self.critic_tx.update(
            critic_grads, self.critic_opt_state, self.critic_params
        )
        
        critic_params = optax.apply_updates(self.critic_params, critic_updates)

        return self.replace(
            step=self.step + 1,
            params=actor_params,
            critic_params=critic_params,
            opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            actor_batch_stats=actor_batch_stats or self.actor_batch_stats,
            critic_batch_stats=critic_batch_stats or self.critic_batch_stats,
        )


def initialize_networks(
    action_dim: int,
    observation_dim: int,
    actor_dense_dim: int,
    critic_dense_dim: int,
    actor_lr: float,
    critic_lr: float,
    max_grad_norm: float = 0.5,
) -> Tuple[ActorNetwork, CriticNetwork, ActorCriticTrainState]:
    """Initializes the Actor-Critic Networks and the custom Training State object!

    Args:

        action_dim (int): The dimensions of the Action Space; needed by the Actor Network
        observation_dim (int): The dimensions of the Observation Space; needed for layer initialization
        actor_dense_dim (int): The out dimensions of the Dense layers in the Actor Network
        critic_dense_dim (int): The out dimensions of the Dense layers in the Critic Network
        actor_lr (float): The learning rate for the Actor Network
        critic_lr (float): The learning rate for the Critic Network
        max_grad_norm (float): The maximum gradient norm for the networks

    Returns:

        Tuple [ActorNetwork, CriticNetwork, ActorCriticTrainState]:

            - ActorNetwork: The designated Actor Network
            - CriticNetwork: The designated Critic Network
            - ActorCriticTrainState: The initialized training state
    """

    # Initializing the networks
    actor_network = ActorNetwork(action_dim=action_dim, dense_dim=actor_dense_dim)
    critic_network = CriticNetwork(dense_dim=critic_dense_dim)

    # Defining PRNG keys for randomized weight initialization
    master_rng_key = jax.random.PRNGKey(0)
    actor_key, critic_key = jax.random.split(master_rng_key)

    dummy_observation = jnp.ones((1, observation_dim))

    # Fetching the Actor-Critic metadata
    actor_metadata = actor_network.init(actor_key, dummy_observation)
    critic_metadata = critic_network.init(critic_key, dummy_observation)

    actor_params = actor_metadata["params"]
    actor_batch_stats = actor_metadata.get("batch_stats", None)

    critic_params = critic_metadata["params"]
    critic_batch_stats = critic_metadata.get("batch_stats", None)

    # Creating & initializing the optimizers for both networks!
    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adamw(learning_rate=actor_lr)
    )

    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm), optax.adamw(learning_rate=critic_lr)
    )

    actor_opt_state = actor_tx.init(actor_params)
    critic_opt_state = critic_tx.init(critic_params)

    # Creating the custom training state now
    training_state = ActorCriticTrainState(
        step=0,
        params=actor_params,
        critic_params=critic_params,
        opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,
        apply_fn=actor_network.apply,
        critic_apply_fn=critic_network.apply,
        tx=actor_tx,
        critic_tx=critic_tx,
        actor_batch_stats=actor_batch_stats,
        critic_batch_stats=critic_batch_stats,
    )

    return actor_network, critic_network, training_state


##############################################
#
# 3. Defining utility calculation functions
#
##############################################


def GAE_function(rewards, values, dones, gamma=0.99, lam=0.95):
    """Calculates the Generalized Advantage Estimation (GAE)
    for the given rewards, values, dones, gamma and lambda.

    Keeping in mind, the rewards will have one lesser value than the values array!
    This is because the values array will have the val pred for the terminal state too :)

    Args:
        rewards (jnp.ndarray): A Jax-Numpy array denoting the rewards received across the rocket trajectory
        values (jnp.ndarray): A Jax-Numpy array denoting the values predicted by the Critic Network
        dones (jnp.ndarray): A boolean array denoting whether an episode ends at this timestep
        gamma (float32): A float value denoting the discount factor for future values
        lam (float32): A float value denoting the trustworthiness of future values

    Returns:
        jnp.ndarray: A Jax-Numpy array denoting the GAE values
    """

    # Calculating the TD values (delta) for all the timesteps! :)
    td = rewards + gamma * values[:, 1:] * (1 - dones) - values[:, :-1]

    # GAE is calculated from the last timestep to the first, so reversing!
    td_reversed = jnp.flip(td, axis=1).T
    dones_reversed = jnp.flip(dones, axis=1).T

    def scan_fn(carry, X):
        """Helper function to calculate GAE with JAX-supported vectorized formats

        Args:
            carry (jnp.array): The prior GAE values for the batches. Shape: [Batch Size]
            X (jnp.array): The TD and Done values at step T. Each of Shape: [Batch Size]

        Returns:
            jnp.array: Returns the GAE values for the timestep T - 1. Shape: [Batch Size]
        """

        td_t, done_t = X

        gae_t = td_t + gamma * lam * (1 - done_t) * carry

        return gae_t, gae_t

    # Defining the carry values across all batches (will be 0 for the beginning)
    carries = jnp.zeros(rewards.shape[0])

    # Applying the scan function, and flipping again to gain original temporal order!
    _, gae_reversed = jax.lax.scan(scan_fn, carries, (td_reversed, dones_reversed))

    advantages = jnp.flip(gae_reversed.T, axis=0)

    return advantages
