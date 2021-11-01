import os

from numpy.core.fromnumeric import argmax
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, input_dims,alpha=1e-4, beta=1e-3, 
            env=None, gamma=0.99, n_actions=1, max_size=1000000, tau=0.005,
            layer1_size=128,  batch_size=10, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, 1)
        self.batch_size = batch_size
        self.n_actions_c = 1
        self.n_actions_a = 3
        self.actor = ActorNetwork(alpha, input_dims=6, n_actions=self.n_actions_a,name='actor')
        self.critic_1 = CriticNetwork(beta=beta, input_dims=6, n_actions=self.n_actions_c,
                    name='critic_1')
        #self.critic_2 = CriticNetwork(beta, T.tensor([6,]), n_actions=n_actions,name='critic_2')
        self.value = ValueNetwork(beta=beta, input_dims=6, name='value')
        self.target_value = ValueNetwork(beta=beta, input_dims=6, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        observation_a=observation[:6]
        observation_c=observation[5:]
        state = T.Tensor(observation).to(self.actor.device)
        #print(state.shape)
        actions, lp,p = self.actor.sample_normal(state[:6], reparameterize=False)
        a=T.nn.Softmax(dim=0)
        b=a(T.tensor(lp.cpu().detach().numpy()))
        print(lp)
        b=T.argmax(b)-1
        print(b)
        return b.item()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        #self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        #self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        #print('\n\n\n\nLEARNING\n\n\n\n')
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        #print('grappa')
        #print(action)
        #print(state)

        value = self.value(state[:,5:]).view(-1)
        valueo_ = self.target_value(state_[:,5:])
        value_=valueo_.view(-1)
        i=0
        for el in done:
            if el ==True:
                value_[i] = 0.0
            i+=1

        actions, log_probs,p = self.actor.sample_normal(state[:,:6], reparameterize=False)
        lp = log_probs
        #print(lp)
        #lp=lp.view(-1)
        #print(lp)
        a=T.nn.Softmax(dim=0)
        b=a(lp)
        
        b=T.argmax(b,axis=1)-1
        q1_new_policy = self.critic_1.forward(state[:,:6], b.view(-1,1))
        #q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = q1_new_policy
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs.view(-1)
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs,p = self.actor.sample_normal(state[:,:6], reparameterize=True)
        lp = log_probs
        a=T.nn.Softmax(dim=0)
        b=a(lp)
        b=T.argmax(b,axis=1)-1
        q1_new_policy = self.critic_1.forward(state[:,:6], b.view(-1,1))
        #q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = q1_new_policy
        #critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs.view(-1) - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        #self.critic_2.optimizer.zero_grad()
        #print('hi')
        #print((self.scale*reward).shape)
        #print((self.gamma*valueo_).shape)
        q_hat = self.scale*reward.view(-1,1) + self.gamma*valueo_
        q1_old_policy = self.critic_1.forward(state[:,5:], action).view(-1)
        #q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat.view(-1))
        #critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss #+ critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        #self.critic_2.optimizer.step()

        self.update_network_parameters()
