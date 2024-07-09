import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

def trainDQN(q_net,
             q_target,
             memory,
             optimizer,
             batch_size,
             gamma):
    # ! We sample from the same Replay Buffer n=10 times
    for _ in range(10):
        # ! Monte Carlo sampling of a batch
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # ! Get the Q-values
        q_out = q_net(s)

        # ! DQN update rule
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ReplayBufferConnectGame():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class QConnectNet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(QConnectNet, self).__init__()
        self.fc1 = nn.Linear(no_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, state, epsilon, possible_actions):
        a = self.forward(state)
        if random.random() < epsilon:
            return random.choice(possible_actions)
        else:
            mask = torch.full(a.size(), float('-inf'))
            mask[possible_actions] = 0
            a = a + mask
            return a.argmax().item()
