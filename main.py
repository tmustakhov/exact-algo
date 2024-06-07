import time

from env import Env, DataGenerator
from env_reachable import Env as Env_reachable
from exact_algo import RandomAgent, MaxReachable

args = {
    'n_epochs': 1,
    'n_batch': 5,
    'batch_size': 512,
    'n_nodes': 10,
    'initial_demand_size': 2,
    'max_load': 9,
    'speed': 0.5,
    'lambda': 1,
    'data_dir': 'datasets',
    'log_dir': 'logs',
    'save_path': 'saved_models',
    'decode_len': 10,
    'actor_net_lr': 0.0001,
    'lr_decay': 1.0,
    'max_grad_norm': 1.0,
    'save_interval': 1,
    'bl_alpha': 0.05,
    'embedding_dim': 128,

}
data_generator = DataGenerator(args)
data = data_generator.get_test_all()
env = Env(args)
env_reachable = Env_reachable(args)
random = RandomAgent(env, args)
reachable = MaxReachable(env_reachable, args)
start = time.time()
print(random.solve(data))
print(time.time() - start, "s")
start = time.time()
print(random.solveMaxDemand(data))
print(time.time() - start, "s")
start = time.time()
print(reachable.solveMaxReachable(data))
print(time.time() - start, "s")
