import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

class DQN_Agent:
    """Deep Q-Network agent."""
    def __init__(self, state_size, action_size, gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma            # Discount factor
        self.epsilon = 1.0            # Exploration rate
        self.epsilon_min = 0.05       # Min exploration rate
        self.epsilon_decay = 0.995    # Decay rate
        self.update_rate = 10         # Target update frequency

        # Q-network and target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    def _build_model(self):
        """Build CNN for Q-learning."""
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='huber', optimizer=Adam(learning_rate=0.00025))
        return model

    def act(self, state):
        """Choose action (epsilon-greedy)."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def greedy_act(self, state):
        """Choose best action (no exploration)."""
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, memory, batch_size):
        """Train model with replay memory."""
        states, actions, rewards, next_states, done_flags = memory.sample(batch_size)
        target_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = rewards[i] if done_flags[i] else rewards[i] + self.gamma * np.max(next_q[i])
            target_q[i][actions[i]] = target

        history = self.model.fit(states, target_q, epochs=1, verbose=0)
        loss = history.history['loss'][-1]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def update_target_model(self):
        """Copy weights to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)
