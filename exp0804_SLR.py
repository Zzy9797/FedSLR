import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
import argparse
import gc
import itertools
import numpy as np
import os
import sys
import time
import pickle
from copy import deepcopy
from tqdm import tqdm
import warnings
from datasets import get_dataset
import exp0804_SLR_model
from exp0804_SLR_model import all_models, initialize_mask
import logging
from logging import handlers
import matplotlib.pyplot as plt
import wandb
import random
rng = np.random.default_rng()
parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, help='learning rate', default=0.01)
parser.add_argument('--clients',
                    type=int,
                    help='number of clients per round',
                    default=20)
parser.add_argument('--rounds',
                    type=int,
                    help='number of global rounds',
                    default=1000)
parser.add_argument('--epochs',
                    type=int,
                    help='number of local epochs',
                    default=10)
parser.add_argument('--dataset',
                    type=str,
                    choices=('mnist', 'emnist', 'cifar10', 'cifar100'),
                    default='cifar100',
                    help='Dataset to use')
parser.add_argument('--distribution',
                    type=str,
                    choices=('dirichlet', 'dirichlet_hetero', 'iid'),
                    default='dirichlet_hetero',
                    help='how should the dataset be distributed?')
parser.add_argument(
    '--beta',
    type=float,
    default=0.1,
    help='Beta parameter (unbalance rate) for Dirichlet distribution')
parser.add_argument(
    '--total-clients',
    type=int,
    help='split the dataset between this many clients. Ignored for EMNIST.',
    default=200)
parser.add_argument('--sparsity',
                    type=float,
                    default=0.5,
                    help='sparsity from 0 to 1')
parser.add_argument('--rate-decay-method',
                    default='cosine',
                    choices=('constant', 'cosine'),
                    help='annealing for readjustment ratio')
parser.add_argument('--rate-decay-end',
                    default=None,
                    type=int,
                    help='round to end annealing')
parser.add_argument('--readjustment-ratio',
                    type=float,
                    default=0.005,
                    help='readjust this many of the weights each time')
parser.add_argument('--pruning-begin',
                    type=int,
                    default=9,
                    help='first epoch number when we should readjust')
parser.add_argument('--pruning-interval',
                    type=int,
                    default=10,
                    help='epochs between readjustments')
parser.add_argument('--rounds-between-readjustments',
                    type=int,
                    default=10,
                    help='rounds between readjustments')
parser.add_argument('--sparsity-distribution',
                    default='erk',
                    choices=('uniform', 'er', 'erk'))
parser.add_argument(
    '--final-sparsity',
    type=float,
    default=None,
    help=
    'final sparsity to grow to, from 0 to 1. default is the same as --sparsity'
)
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='local client batch size')
parser.add_argument('--l2',
                    default=0.001,
                    type=float,
                    help='L2 regularization strength')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='Local client SGD momentum parameter')
parser.add_argument('--device', default='0', type=str)
parser.add_argument('--min-votes',
                    default=0,
                    type=int,
                    help='Minimum votes required to keep a weight')
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help='upload as fp16')
parser.add_argument('--wandb', default=False)
parser.add_argument('--shuffle', default=True)
parser.add_argument('--evaluate-whole', default=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
if args.rate_decay_end is None:
    args.rate_decay_end = args.rounds // 2
if args.final_sparsity is None:
    args.final_sparsity = args.sparsity

def get_time():
    now = int(time.time())
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d_%H:%M:%S", timeArray)
    print(otherStyleTime)
    return otherStyleTime

def result_log(time_path, log_acc, log_round, log_upload, log_bestacc, *arg,
               **kwargs):
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.title('accuracy_round')
    plt.plot(log_round, log_acc)
    plt.savefig(time_path + '/accuracy_round.jpg')
    plt.clf()
    plt.xlabel('upload')
    plt.ylabel('accuracy')
    plt.title('accuracy_upload')
    plt.plot(log_upload, log_acc)
    plt.savefig(time_path + '/accuracy_upload.jpg')
    plt.clf()
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.title('best_accuracy_round')
    plt.plot(log_round, log_bestacc)
    plt.savefig(time_path + '/best_accuracy_round.jpg')
    plt.clf()

def print2(time_path, *arg, **kwargs):
    print(*arg,
          **kwargs,
          file=open(os.path.join(time_path, 'log.log'), 'a', encoding='ascii'))
    print(*arg, **kwargs)

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()

def evaluate_seperate(clients, global_model, progress=False):
    with torch.no_grad():
        accuracies = {}
        sparsities = {}
        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()
        for client_id, client in enumerator:
            accuracies[client_id] = client.test(model=global_model).item()
            sparsities[client_id] = client.sparsity()
    return accuracies, sparsities

def evaluate_all(model, test_data):
    correct = 0.
    total = 0.
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=-1)
            correct += sum(labels == outputs)
            total += len(labels)
    return correct.item() / total

class Client:
    def __init__(self,
                 id,
                 train_data,
                 test_data,
                 net=models08.exp0804_SLR_model.MNISTNet,
                 local_epochs=10,
                 learning_rate=0.01):
        self.id = id
        self.train_data, self.test_data = train_data, test_data
        self.net = net().cuda()
        initialize_mask(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.reset_optimizer()
        self.local_epochs = local_epochs
        self.curr_epoch = 0
        self.initial_global_params = None

    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=self.learning_rate,
                                         momentum=args.momentum,
                                         weight_decay=args.l2)

    def reset_weights(self, *args, **kwargs):
        return self.net.reset_weights(*args, **kwargs)

    def sparsity(self, *args, **kwargs):
        return self.net.sparsity(*args, **kwargs)

    def train_size(self):
        return sum(len(x) for x in self.train_data)

    def train(self,
              global_params=None,
              initial_global_params=None,
              readjustment_ratio=0.5,
              readjust=False,
              sparsity=args.sparsity):

        ul_cost = 0
        dl_cost = 0
        mask_changed = self.reset_weights(global_state=global_params,
                                            use_global_mask=True)
        self.reset_optimizer()
        if not self.initial_global_params:
            self.initial_global_params = initial_global_params
        else:
            if mask_changed:
                dl_cost += (1 -self.net.sparsity()) * self.net.param_size+self.net.mask_size
            else:
                dl_cost += (1 -self.net.sparsity()) * self.net.param_size
        self.net.train()
        for epoch in range(self.local_epochs):
            running_loss = 0.
            if args.shuffle:
                random.shuffle(self.train_data)
            for inputs, labels in self.train_data:
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.reset_weights()
                running_loss += loss.item()
            if (self.curr_epoch - args.pruning_begin
                ) % args.pruning_interval == 0 and readjust:
                prune_sparsity = sparsity + (1 - sparsity) * readjustment_ratio
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                self.criterion(outputs, labels).backward()
                self.net.prune_and_grow(prune_sparsity=prune_sparsity,grow_sparsity=sparsity,sparsity_distribution=args.sparsity_distribution)
                ul_cost += self.net.mask_size
            self.curr_epoch += 1
        if args.fp16:
            ul_cost += (1 - self.net.sparsity()) * self.net.param_size
        else:
            ul_cost += (1 - self.net.sparsity()) * self.net.param_size 
        ret = dict(state=self.net.state_dict(),
                   dl_cost=dl_cost,
                   ul_cost=ul_cost)
        return ret

    def test(self, model=None):
        correct = 0.
        total = 0.
        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.cuda()
        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_data):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)
        if model is not _model:
            del _model
        return correct / total

if __name__ == '__main__':

    otherStyleTime = get_time()
    otherStyleTime = 'exp0804_SLR_' + otherStyleTime
    time_path = './log08/' + otherStyleTime
    os.makedirs(time_path)
    print2(time_path, args)
    print('Fetching dataset...')
    loaders, whole_test_data = get_dataset(args.dataset,
                                           clients=args.total_clients,
                                           mode=args.distribution,
                                           beta=args.beta,
                                           batch_size=args.batch_size)
    print('Initializing clients...')
    clients = {}
    client_ids = []
    for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
        cl = Client(client_id,
                    *client_loaders,
                    net=all_models[args.dataset],
                    learning_rate=args.eta,
                    local_epochs=args.epochs)
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()
    global_model = all_models[args.dataset]().cuda()
    initialize_mask(global_model)
    global_model.layer_prune(sparsity=args.sparsity,
                             sparsity_distribution=args.sparsity_distribution
                             )
    zero=0
    for name, layer in global_model.named_children():
            for pname, param in layer.named_parameters():

                zero+=torch.count_nonzero(param.data.flatten())
    initial_global_params = deepcopy(global_model.state_dict())
    download_cost = np.zeros(len(clients))
    upload_cost = np.zeros(len(clients))
    best_acc = 0
    acc_list = []
    round_list = []
    best_acc_list = [] 
    upload_cost_list = []
    total_upload = 0
    for server_round in tqdm(range(args.rounds)):
        round_list.append(server_round + 1)
        if (server_round-1) %100 ==0:
            os.makedirs(os.path.join(time_path,'round'+str(server_round)))
        client_indices = rng.choice(list(clients.keys()),
                                    size=args.clients)

        global_params = global_model.state_dict()
        aggregated_params = {}
        aggregated_masks = {}
        for name, param in global_params.items():
            if name.endswith('_mask'):
                continue
            aggregated_params[name] = torch.zeros_like(
                param, dtype=torch.float).cuda()
            if name.endswith('weight'):
                aggregated_masks[name] = torch.zeros_like(param).cuda()
        total_sampled = 0
        for client_id in client_indices:
            client = clients[client_id]
            i = client_ids.index(client_id)
            t0 = time.process_time()
            if args.rate_decay_method == 'cosine':
                readjustment_ratio = global_model._decay(
                    server_round,
                    alpha=args.readjustment_ratio,
                    t_end=args.rate_decay_end)
            else:
                readjustment_ratio = args.readjustment_ratio
            readjust = (
                server_round - 1
            ) % args.rounds_between_readjustments == 0 and readjustment_ratio > 0.
            if readjust:
                print('readjusting', readjustment_ratio)
            if server_round <= args.rate_decay_end:
                round_sparsity = args.sparsity * (
                    args.rate_decay_end - server_round
                ) / args.rate_decay_end + args.final_sparsity * server_round / args.rate_decay_end
            else:
                round_sparsity = args.final_sparsity
            train_result = client.train(
                global_params=global_params,
                initial_global_params=initial_global_params,
                readjustment_ratio=readjustment_ratio,
                readjust=readjust,
                sparsity=round_sparsity
                )
            cl_params = train_result['state']
            download_cost[i] = train_result['dl_cost']
            upload_cost[i] = train_result['ul_cost']
            t1 = time.process_time()
            client.net.clear_gradients()
            cl_weight_params = {}
            cl_mask_params = {}
            for name, cl_param in cl_params.items():
                if name.endswith('_orig'):
                    name = name[:-5]
                elif name.endswith('_mask'):
                    name = name[:-5]
                    cl_mask_params[name] = cl_param
                    continue
                cl_weight_params[name] = cl_param
                if args.fp16:
                    cl_weight_params[name] = cl_weight_params[name].to(
                        torch.bfloat16).to(torch.float)
            for name, cl_param in cl_weight_params.items():
                if name in cl_mask_params:
                    cl_mask = cl_mask_params[name]
                    aggregated_params[name].add_(client.train_size() *
                                                 cl_param)  
                    aggregated_masks[name].add_(client.train_size() * cl_mask)
                else:
                    aggregated_params[name].add_(client.train_size() *
                                                 cl_param)
        for name, param in aggregated_params.items():
            aggregated_params[name] /= sum(clients[i].train_size()
                                               for i in client_indices)
            if name in aggregated_masks:
                aggregated_masks[name] = F.threshold_(aggregated_masks[name],
                                                  args.min_votes, 0)
        global_model.load_state_dict(aggregated_params,strict=False)
        global_model.layer_prune_agg(
                sparsity=round_sparsity,
                sparsity_distribution=args.sparsity_distribution,agg_mask=aggregated_masks)
        torch.cuda.empty_cache()
        accuracy = evaluate_all(global_model, whole_test_data)
        accuracies, sparsities = evaluate_seperate(clients,
                                                   global_model,
                                                   progress=True)
        accuracies = [value for key, value in accuracies.items()]
        sparsities = [value for key, value in sparsities.items()]
        if args.evaluate_whole:
            print2(time_path, 'Server Accuracy: {}'.format(accuracy))
            acc_list.append(accuracy)
            best_acc = max(best_acc, accuracy)
        else:
            print2(time_path,
                   'Average Accuracy: {}'.format(np.mean(accuracies)))
            acc_list.append(np.mean(accuracies))
            best_acc = max(best_acc, np.mean(accuracies))

        print2(time_path,
            f'SPARSITY: mean={np.mean(sparsities)}, std={np.std(sparsities)}, min={np.min(sparsities)}, max={np.max(sparsities)}'
        )
        print2(time_path, 'Best Accuracy: {}'.format(best_acc))
        best_acc_list.append(best_acc)

        total_upload = total_upload + np.sum(upload_cost) / 8589934592
        print2(time_path, 'Total Upload: {} GiB'.format(total_upload))
        upload_cost_list.append(total_upload)

        if server_round == 0:
            for client_id in clients:
                clients[client_id].initial_global_params = initial_global_params
        if args.wandb:
            log_acc = accuracy if args.evaluate_whole else np.mean(accuracies)
            wandb.log({"acc": log_acc, "upload_cost": total_upload, "bestacc":best_acc})
        download_cost[:] = 0
        upload_cost[:] = 0
        result_log(time_path, acc_list, round_list, upload_cost_list,
                   best_acc_list)
    if args.wandb:
        wandb.finish()
