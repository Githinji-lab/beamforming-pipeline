import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

# --- SAC Networks ---
def create_sac_actor_network(state_dim, action_dim, log_std_min=-20.0, log_std_max=2.0):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    mean = layers.Dense(action_dim, activation="linear")(x)
    log_std_dev = layers.Dense(action_dim, activation="linear")(x)
    log_std_dev = keras.ops.clip(log_std_dev, log_std_min, log_std_max)
    return keras.Model(inputs, [mean, log_std_dev])

def create_sac_critic_network(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    state_out = layers.Dense(512, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(256, activation="leaky_relu")(state_out)
    action_input = layers.Input(shape=(action_dim,))
    action_out = layers.Dense(256, activation="relu")(action_input)
    concat = layers.Concatenate()([state_out, action_out])
    x = layers.Dense(256, activation="leaky_relu")(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model([state_input, action_input], outputs)

# --- SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, scaler_X, scaler_y,
                 gamma=0.99, tau=0.005, actor_lr=0.0003, critic_lr=0.0003, alpha_lr=0.0003,
                 reward_scale=1.0, target_entropy=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.replay_buffer = replay_buffer
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        self.actor_model = create_sac_actor_network(state_dim, action_dim)
        self.target_actor = create_sac_actor_network(state_dim, action_dim)
        self.target_actor.set_weights(self.actor_model.get_weights())

        self.critic_q1 = create_sac_critic_network(state_dim, action_dim)
        self.critic_q2 = create_sac_critic_network(state_dim, action_dim)
        self.target_critic_q1 = create_sac_critic_network(state_dim, action_dim)
        self.target_critic_q2 = create_sac_critic_network(state_dim, action_dim)
        self.target_critic_q1.set_weights(self.critic_q1.get_weights())
        self.target_critic_q2.set_weights(self.critic_q2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_q1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic_q2_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.target_entropy = tf.constant(target_entropy, dtype=tf.float32) if target_entropy is not None else -tf.constant(action_dim, dtype=tf.float32)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.alpha = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.update_alpha()

    @tf.function
    def _sample_action_from_actor(self, states_input):
        mean, log_std_dev = self.actor_model(states_input)
        std_dev = tf.exp(log_std_dev)
        distribution = tfp.distributions.Normal(loc=mean, scale=std_dev)
        raw_actions = distribution.sample()
        actions = tf.math.tanh(raw_actions)
        log_prob = distribution.log_prob(raw_actions)
        log_prob -= tf.reduce_sum(tf.math.log(tf.clip_by_value(1 - actions**2, 1e-6, 1.0)), axis=1)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        return actions, log_prob

    def choose_action(self, state, evaluate=False):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        if evaluate:
            mean, _ = self.actor_model(state_tensor)
            return tf.math.tanh(mean[0]).numpy()
        else:
            action, _ = self._sample_action_from_actor(state_tensor)
            return action.numpy()[0]

    @tf.function
    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size: return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.expand_dims(tf.convert_to_tensor(reward_batch, dtype=tf.float32), axis=1)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.bool)

        with tf.GradientTape(persistent=True) as tape:
            current_actions, current_log_probs = self._sample_action_from_actor(state_batch)
            alpha_loss = -tf.math.reduce_mean(self.log_alpha * (current_log_probs + self.target_entropy))

            next_actions, next_log_probs = self._sample_action_from_actor(next_state_batch)
            target_q1 = self.target_critic_q1([next_state_batch, next_actions])
            target_q2 = self.target_critic_q2([next_state_batch, next_actions])
            min_target_q = tf.minimum(target_q1, target_q2)
            target_q = reward_batch * self.reward_scale + self.gamma * (1 - tf.cast(done_batch, tf.float32)) * (min_target_q - self.alpha * next_log_probs)

            current_q1 = self.critic_q1([state_batch, action_batch])
            current_q2 = self.critic_q2([state_batch, action_batch])
            critic_q1_loss = tf.math.reduce_mean(tf.math.square(target_q - current_q1))
            critic_q2_loss = tf.math.reduce_mean(tf.math.square(target_q - current_q2))
            critic_loss = critic_q1_loss + critic_q2_loss

            q1_actor = self.critic_q1([state_batch, current_actions])
            q2_actor = self.critic_q2([state_batch, current_actions])
            min_q_actor = tf.minimum(q1_actor, q2_actor)
            actor_loss = tf.math.reduce_mean(self.alpha * current_log_probs - min_q_actor)

        critic_q1_grad = tape.gradient(critic_q1_loss, self.critic_q1.trainable_variables)
        self.critic_q1_optimizer.apply_gradients(zip(critic_q1_grad, self.critic_q1.trainable_variables))

        critic_q2_grad = tape.gradient(critic_q2_loss, self.critic_q2.trainable_variables)
        self.critic_q2_optimizer.apply_gradients(zip(critic_q2_grad, self.critic_q2.trainable_variables))

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        self.update_alpha()

    @tf.function
    def update_alpha(self):
        self.alpha.assign(tf.exp(self.log_alpha))

    @tf.function
    def update_target_networks(self):
        for w, target_w in zip(self.critic_q1.variables, self.target_critic_q1.variables):
            target_w.assign(self.tau * w + (1.0 - self.tau) * target_w)
        for w, target_w in zip(self.critic_q2.variables, self.target_critic_q2.variables):
            target_w.assign(self.tau * w + (1.0 - self.tau) * target_w)
        for w, target_w in zip(self.actor_model.variables, self.target_actor.variables):
            target_w.assign(self.tau * w + (1.0 - self.tau) * target_w)

# --- FIX FOR SACAgent alpha_optimizer ERROR ---
def patch_sac_agent(sac_agent, alpha_lr=0.0001):
    if not hasattr(sac_agent, 'alpha_optimizer'):
        sac_agent.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)
        print("Added alpha_optimizer to SACAgent instance.")
    else:
        print("SACAgent already has alpha_optimizer.")

class SACAgentFixed(SACAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'alpha_optimizer'):
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get('alpha_lr', 0.0001))