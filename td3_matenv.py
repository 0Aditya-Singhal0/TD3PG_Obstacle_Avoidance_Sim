import numpy as np
import tensorflow as tf
import keras as keras
from keras.layers import Dense
from keras.optimizers import Adam
import os
import math

# THIS IS THE COPY OF TD3PG CODE HARDCODED FOR MATPLOTLIB ENVIRONMENT...
# CHANGES WILL BE MADE SO THAT IT TAKES ALL THE INFORMATION FROM THE ENV VARIABLE LATER.

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)

        q = self.q(q1_action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name,chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='softplus')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu


class Agent:
    def __init__(self, alpha, beta, input_dims,env,tau,
                 gamma=0.99, update_actor_interval=2, warmup=200,
                 n_actions=7, max_size=1000000, layer1_size=512,
                 layer2_size=256, batch_size=128, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = math.pi/4
        self.min_action = -math.pi/4
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(layer1_size, layer2_size,n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(layer1_size, layer2_size,name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size,name='critic_2')
        self.target_actor = ActorNetwork(layer1_size, layer2_size,n_actions=n_actions,name='target_actor')
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size,name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size,name='target_critic_2')
        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha),loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta),loss='mean_squared_error')
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        # returns a batch size of 1, want a scalar array
        mu = self.actor(state)
        if self.time_step > self.warmup:
            mu += np.random.normal(scale=self.noise, size=(self.n_actions,))     
        mu_prime = mu 

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1
        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, self.min_action,self.max_action)

            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)
            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)


            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma*critic_value_*(dones)
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tape1.gradient(critic_1_loss,
                                          self.critic_1.trainable_variables)
        critic_2_gradient = tape2.gradient(critic_2_loss,
                                          self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter == 0:
            with tf.GradientTape() as tape3:
                new_actions = self.actor(states)
                critic_1_value = self.critic_1(states, new_actions)
                actor_loss = -tf.math.reduce_mean(critic_1_value)

            actor_gradient = tape3.gradient(actor_loss,
                                           self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights1 = []
        targets1 = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights1.append(weight * tau + targets1[i]*(1-tau))
        self.target_actor.set_weights(weights1)

        weights2 = []
        targets2 = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights2.append(weight * tau + targets2[i]*(1-tau))
        self.target_critic_1.set_weights(weights2)

        weights3 = []
        targets3 = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights3.append(weight * tau + targets3[i]*(1-tau))
        self.target_critic_2.set_weights(weights3)
        
        
    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)
        
        with open('tmp/replay_buffer_memory.npy', 'wb') as f:
            np.save(f, self.memory.state_memory)
            np.save(f,self.memory.new_state_memory)
            np.save(f,self.memory.action_memory)
            np.save(f,self.memory.reward_memory)
            np.save(f,self.memory.terminal_memory)

        
    def load_models(self):
        print('... trying to load models ...')
        path = self.actor.checkpoint_dir
        if not (self.isEmpty(path)):
            self.actor.load_weights(self.actor.checkpoint_file).expect_partial()
            self.critic_1.load_weights(self.critic_1.checkpoint_file).expect_partial()
            self.critic_2.load_weights(self.critic_2.checkpoint_file).expect_partial()
            self.target_actor.load_weights(self.target_actor.checkpoint_file).expect_partial()
            self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file).expect_partial()
            self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file).expect_partial()
            with open('tmp/replay_buffer_memory.npy', 'rb') as f: 
                self.memory.state_memory=np.load(f)
                self.memory.new_state_memory=np.load(f)
                self.memory.action_memory=np.load(f)
                self.memory.reward_memory=np.load(f)
                self.memory.terminal_memory=np.load(f)
            self.memory.mem_cntr = len(self.memory.action_memory)
        return
    
    def isEmpty(self, path):
        check=True
        
        if os.path.exists(path) and not os.path.isfile(path):
            # Checking if the directory is empty or not
            if not os.listdir(path):
                print("No checkpoints found")
            else:
                print("Model found")
                check = False
        else:
            print("The path is either for a file or not valid")
            
        return check
            