import flax.linen as nn
import jax.numpy as jnp
import jax

class ActorNetwork(nn.Module):

    action_dim: int
    
    def setup(self):
        """Defines the Policy network which outputs
        the logits for each action in the action space.
        """
        
        # One point to note --> 256 is the output dimension!
        # The input dimension is inferred during the first forward pass by Flax.
        
        # I found this to be a very cool mechanism :)
        self.dense1 = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())
        self.dense2 = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())
        
        self.batch_norm_1 = nn.BatchNorm()
        self.batch_norm_2 = nn.BatchNorm()
        
        # Gives the logits for each action in the action space
        self.logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.01))

    def __call__(self, x, training: bool = True) -> jnp.ndarray:
        """Performs the forward pass given an input JAX ndarray X.

        Args:
            x (jnp.ndarray): A JAX-Numpy ndarray denoting the input [batch_size, observation_dim]
            training (bool): A bool value indicating whether the model is in training mode. Effects batch norm!
        
        Returns:
            jnp.ndarray: The output tensor of shape [batch_size, action_dim]
        """
        
        x = nn.relu(self.dense1(x))
        x = self.batch_norm_1(x, use_running_average=not training)
        
        x = nn.relu(self.dense2(x))
        x = self.batch_norm_2(x, use_running_average=not training)
        
        return self.logits(x)

class CriticNetwork(nn.Module):
    
    def setup(self):
        """Defines the value network which estimates
        the estimated values of given states.
        """
        
        self.dense1 = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())
        self.dense2 = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())
        
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

def GAE_function(rewards, values, dones, gamma_val, lambda_val):
    """Calculates the Generalized Advantage Estimation (GAE)
    for the given rewards, values, dones, gamma and lambda.

    Args:
        rewards (jnp.ndarray): A Jax-Numpy array denoting the rewards received across the rocket trajectory
        values (jnp.ndarray): A Jax-Numpy array denoting the values predicted by the Critic Network
        dones (jnp.ndarray): A boolean array denoting whether an episode ends at this timestep
        gamma_val (float32): A float value denoting the discount factor for future values
        lambda_val (float32): A float value denoting the trustworthiness of future values 

    Returns:
        jnp.ndarray: A Jax-Numpy array denoting the GAE values
    """
    
    deltas = (rewards + 1.0 - dones) * gamma_val * values[1:] - values[:-1]
    
     # Compute GAE recursively using JAX's scan
    def gae_step(carry, x):
    
        advantage, discount = carry
        delta, done = x
    
        # Update advantage using GAE formula
        advantage = delta + (1.0 - done) * gamma_val * lambda_val * advantage
        return (advantage, discount), advantage
    
    # Initialize with zeros and scan backwards
    _, advantages = jax.lax.scan(
        gae_step,
        (0.0, 1.0),  # Initial advantage and discount
        (deltas, dones),
        reverse=True
    )
    
    return advantages