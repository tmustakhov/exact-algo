import sys

import torch
import random
import os
import math


def create_test_dataset(args):
    batch_size = args['batch_size']
    n_nodes = args['n_nodes']
    data_dir = args['data_dir']
    # build task name and datafiles
    task_name = 'VRP-size-{}-len-{}.txt'.format(batch_size, n_nodes)
    fname = os.path.join(data_dir, task_name)

    # create/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = torch.load(fname)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size batch_size
        input_data = torch.rand(batch_size, n_nodes, 2)
        # fix depot (0.5, o.5)
        input_data[:, n_nodes - 1, 0] = 0.5
        input_data[:, n_nodes - 1, 1] = 0.5
        time_demand = generate_events(args)
        data = torch.cat((input_data, time_demand), -1)
        torch.save(data, fname)

    return data


class DataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = args['batch_size']
        self.n_nodes = args['n_nodes']
        # create test data
        self.test_data = create_test_dataset(args)

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self, n_batches):
        train_data = torch.rand(n_batches, self.batch_size, self.n_nodes, 2)
        train_data[:, :, self.n_nodes - 1, 0] = 0.5
        train_data[:, :, self.n_nodes - 1, 1] = 0.5
        time_demand = torch.zeros(n_batches, self.batch_size, self.n_nodes, 4)
        for i in range(n_batches):
            time_demand[i] = generate_events(self.args)
        return torch.cat((train_data, time_demand), -1)

    def get_test_next(self):
        pass

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data


def generate_events(args):
    batch_size = args['batch_size']
    _lambda = args['lambda']
    n_nodes = args['n_nodes']
    max_load = args['max_load']
    initial_demand_size = args['initial_demand_size']

    time_demand = torch.zeros(batch_size, n_nodes, 4)
    # time_demand = torch.zeros(2,10,4)

    for k in range(batch_size):
        _arrival_time = 0
        _inter_arrival_time = 0
        nodes = [i for i in range(n_nodes - 1)]
        for i in range(n_nodes - 1):

            if i >= initial_demand_size:
                # Get the next probability value from Uniform(0,1)
                p = random.random()
                # Plug it into the inverse of the CDF of Exponential(_lamnbda)
                _inter_arrival_time = -math.log(1.0 - p) / _lambda

                # Add the inter-arrival time to the running sum
                _arrival_time = _arrival_time + _inter_arrival_time

            # Choose random node index
            idx = random.choice(nodes)
            nodes.remove(idx)

            time_demand[k][idx][0] = _arrival_time
            time_demand[k][idx][1] = _inter_arrival_time
            time_demand[k][idx][2] = torch.randint(1, max_load, (1, 1))
    return time_demand


class Env(object):
    def __init__(self, args):

        self.max_load = args['max_load']
        self.n_nodes = args['n_nodes']
        self.batch_size = args['batch_size']
        self.speed = args['speed']
        self.initial_demand_size = args['initial_demand_size']
        self.args = args

    def reset(self, data):
        self.input_pnt = data[:, :, :2]
        self.time_demand = data[:, :, 2:].detach().clone()
        self.dist_mat = torch.zeros(self.batch_size, self.n_nodes, self.n_nodes, dtype=torch.float)

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                self.dist_mat[:, i, j] = ((self.input_pnt[:, i, 0] - self.input_pnt[:, j, 0]) ** 2 +
                                          (self.input_pnt[:, i, 1] - self.input_pnt[:, j, 1]) ** 2) ** 0.5
                self.dist_mat[:, j, i] = self.dist_mat[:, i, j]
        self.max_dist = torch.max(torch.max(self.dist_mat, 1)[0], 1)[0]

        # TODO: after fixing leaving of customer waiting time is too small
        self.waiting_time = torch.ones(self.batch_size) * 5
        self.waiting_time = self.waiting_time.reshape((-1, 1)).repeat(1, self.n_nodes)
        self.time_demand[:, :, 3] = self.time_demand[:, :, 0] + self.waiting_time

        self.cur_load = torch.full((self.batch_size, 1), self.max_load, dtype=torch.long)
        self.cur_loc = torch.full((self.batch_size, 1), self.n_nodes - 1)
        self.mask = torch.ones(self.batch_size, self.n_nodes, dtype=torch.long)

        self.demand = torch.zeros(self.batch_size, self.n_nodes)
        # generate random indices for initial demand
        # initial_demand_shape = (self.batch_size, self.initial_demand_size)
        #  idx = (torch.arange(self.initial_demand_size)[None, :]).repeat(self.batch_size, 1)
        #   x = (torch.arange(self.batch_size)[:, None]).repeat(1, self.initial_demand_size)
        # self.demand[x, idx] = torch.randint(1, self.max_load + 1, initial_demand_shape)
        # self.mask[x, idx] = 0
        self.cur_time = torch.zeros(self.batch_size)
        b, n = torch.where(self.time_demand[:, :self.n_nodes - 1, 0] <= self.cur_time[:, None].repeat_interleave(self.n_nodes - 1,axis=1))
        self.demand[b, n] = self.time_demand[b, n, 2]
        self.time_demand[b, n, 2] = 0
        self.reward = torch.zeros(self.batch_size)
        self.total_demand = torch.sum(self.time_demand[:, :, 2], 1)
        data = torch.cat((self.input_pnt, self.demand[:, :, None]), -1)
        self.mask = torch.where(torch.logical_and(self.cur_load >= self.demand, self.demand != 0), 0, 1)

        return data, self.mask, self.demand, self.cur_load

    def step(self, idx):
        idx = idx.view(-1, 1).to(torch.device("cpu"))
        time = self.dist_mat[(torch.arange(self.batch_size))[:, None], self.cur_loc, idx] / self.speed
        time = time.view(-1)

        for b_s in range(self.batch_size):
            # when vehcile waited and clock needs to move by the earliest demand occurance
            if self.cur_loc[b_s] == idx[b_s]:
                min_arr_time = 10000
                for n, event in enumerate(self.time_demand[b_s]):
                    if event[2] != 0 and event[0] <= min_arr_time:
                        min_arr_time = event[0]
                if min_arr_time < 10000:
                    time[b_s] = min_arr_time - self.cur_time[b_s]
                else:
                    for n, event in enumerate(self.time_demand[b_s]):
                        if event[3] > self.cur_time[b_s] and event[3] <= min_arr_time:
                            min_arr_time = event[3]
                    time[b_s] = min_arr_time - self.cur_time[b_s]

        # update demand based on clock

        # release demand if it arrived in this time step
        b, n = torch.where(torch.logical_and(
            self.time_demand[:, :self.n_nodes - 1, 0] > self.cur_time[:, None].repeat_interleave(self.n_nodes - 1,axis=1),
            self.time_demand[:, :self.n_nodes - 1, 0] <= (self.cur_time + time)[:, None].repeat_interleave(self.n_nodes - 1, axis=1)))
        self.demand[b, n] = self.time_demand[b, n, 2]
        self.time_demand[b, n, 2] = 0

        self.cur_time += time

        # update demand if it left the system
        b, n = torch.where(self.time_demand[:, :self.n_nodes - 1, 3] < self.cur_time[:, None].repeat_interleave(self.n_nodes - 1,axis=1))
        self.demand[b, n] = 0

        # update load and reward if demand at idx was not zero, otherwise sat demand is zero
        self.cur_load -= self.demand[(torch.arange(self.batch_size))[:, None], idx].long()

        self.reward[(torch.arange(self.batch_size))[:, None]] += self.demand[
            (torch.arange(self.batch_size))[:, None], idx]

        # update if demand is satisfied, otherwaise the demand there is already zero
        self.demand[(torch.arange(self.batch_size))[:, None], idx] = 0

        self.cur_loc = idx

        # refill if we are in depot
        batch = torch.where(self.cur_loc == self.n_nodes - 1)[0]
        self.cur_load[batch] = self.max_load

        # update mask
        self.mask = torch.where(torch.logical_and(self.cur_load >= self.demand, self.demand != 0), 0, 1)
        for b_s in range(self.batch_size):
            for n in range(self.n_nodes):
                if self.mask[b_s, n] == 0:
                    t = self.dist_mat[b_s][n][self.cur_loc[b_s]] / self.speed
                    if self.time_demand[b_s, n, 3] < self.cur_time[b_s] + t:
                        self.mask[b_s, n] = 1
        # check if sum of mask is equal to n_nodes open depot
        # print("first mask: ", self.mask[0])
        batch = torch.where(torch.sum(self.mask, 1) == self.n_nodes)[0]
        self.mask[batch, self.n_nodes - 1] = 0
        # print("second mask: ", self.mask[0])
        batch = torch.where(self.cur_loc != self.n_nodes - 1)[0]
        self.mask[batch, self.cur_loc[batch, 0]] = 0
        # print("third mask: ", self.mask[0])

        finished = False
        if torch.all(torch.logical_and(torch.sum(self.demand, 1) == 0,
                                       torch.logical_and((self.cur_loc == self.n_nodes - 1).view(-1),
                                                         torch.sum(self.time_demand[:, :, 2], 1) == 0))):
            finished = True

        # concatenate input points with demand
        data = torch.cat((self.input_pnt, self.demand[:, :, None]), -1)

        return data, self.cur_loc, self.mask, self.demand, self.cur_load, finished
