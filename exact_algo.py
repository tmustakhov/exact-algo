import random

import torch


class State(object):

    def __init__(self, batch_size, n_nodes, mask, demand, cur_load):
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.demand = demand
        self.mask = mask
        self.cur_load = cur_load
        self.cur_loc = torch.full((self.batch_size, 1), self.n_nodes - 1)

    def __getitem__(self, item):
        return {
            'cur_loc': self.cur_loc[item],
            'mask': self.mask[item],
            'cur_load': self.cur_load[item],
            'demand': self.demand[item]
        }

    def update(self, cur_loc, mask, demand, cur_load):
        # self.current_node = current_node[:, None]
        self.cur_loc = cur_loc
        self.cur_load = cur_load
        self.demand = demand
        self.mask = mask


class RandomAgent(object):

    def __init__(self, env, args):
        self.env = env
        self.args = args

    def solve(self, data):
        env = self.env
        actions = []
        time_step = 0
        data, mask, demand, cur_load = env.reset(data)

        while time_step < self.args['decode_len']:
            batch, n = torch.where(mask == 0)
            available_nodes = [[] for _ in range(self.args['batch_size'])]
            idx = torch.zeros(self.args['batch_size'], dtype=torch.long)
            for i in range(len(batch)):
                available_nodes[batch[i]].append(n[i])
            for i in range(self.args['batch_size']):
                idx[i] = random.choice(available_nodes[i])
            actions.append(idx)
            time_step += 1
            data, cur_loc, mask, demand, cur_load, finished = env.step(idx)
            if finished:
                break

            # print("{}: {}".format("state update", state[0]))
            # print("{}: {}".format("mask", mask[0]))
            # print("{}: {}".format("state", env.state[0]))

        R = env.reward
        print("R: ", R.mean())
        actions = torch.stack(actions, 1)

        return R, actions

    def solveMaxDemand(self, data):
        env = self.env
        actions = []
        time_step = 0
        data, mask, demand, cur_load = env.reset(data)

        while time_step < self.args['decode_len']:
            # print(mask, 'mask')
            # print(demand, 'demand')
            batch, n = torch.where(mask == 0)
            idx = torch.zeros(self.args['batch_size'], dtype=torch.long)
            max_demand = torch.zeros(self.args['batch_size'], dtype=torch.long)
            for i in range(len(batch)):
                if demand[batch[i]][n[i]] >= max_demand[batch[i]]:
                    idx[batch[i]] = n[i]
                    max_demand[batch[i]] = demand[batch[i]][n[i]]
            # print(idx, 'idx')
            actions.append(idx)
            time_step += 1
            data, cur_loc, mask, demand, cur_load, finished = env.step(idx)
            if finished:
                break

            # print("{}: {}".format("state update", state[0]))
            # print("{}: {}".format("mask", mask[0]))
            # print("{}: {}".format("state", env.state[0]))

        R = env.reward
        print("R: ", R.mean())
        actions = torch.stack(actions, 1)

        return R, actions


class MaxReachable(object):

    def __init__(self, env, args):
        self.env = env
        self.args = args

    def solveMaxReachable(self, data):
        env = self.env
        actions = []
        time_step = 0
        data, mask, demand, cur_load = env.reset(data)

        while time_step < self.args['decode_len']:
            batch, n = torch.where(mask == 0)
            idx = torch.zeros(self.args['batch_size'], dtype=torch.long)
            max_demand = torch.zeros(self.args['batch_size'], dtype=torch.long)
            for i in range(len(batch)):
                if demand[batch[i]][n[i]] >= max_demand[batch[i]]:
                    idx[batch[i]] = n[i]
                    max_demand[batch[i]] = demand[batch[i]][n[i]]
            actions.append(idx)
            time_step += 1
            data, cur_loc, mask, demand, cur_load, finished = env.step(idx)
            if finished:
                break

            # print("{}: {}".format("state update", state[0]))
            # print("{}: {}".format("mask", mask[0]))
            # print("{}: {}".format("state", env.state[0]))

        R = (env.reward)
        print("R: ", R.mean())
        actions = torch.stack(actions, 1)

        return R, actions