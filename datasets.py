import json
import numpy as np
import os
import torch
import torchvision
from tqdm import tqdm



def distribute_clients_categorical(x, p, clients=400, beta=0.1):

    unique, counts = torch.Tensor(x.targets).unique(return_counts=True)

    # Generate offsets within classes
    offsets = np.cumsum(np.broadcast_to(counts[:, np.newaxis], p.shape) * p, axis=1).astype('uint64')


    # Generate offsets for each class in the indices
    inter_class_offsets = np.cumsum(counts) - counts
    
    # Generate absolute offsets in indices for each client
    offsets = offsets + np.broadcast_to(inter_class_offsets[:, np.newaxis], offsets.shape).astype('uint64')
    offsets = np.concatenate([offsets, np.cumsum(counts)[:, np.newaxis]], axis=1).astype('uint64')


    # Use the absolute offsets as slices into the indices
    indices = []
    n_classes_by_client = []
    index_source = torch.LongTensor(np.argsort(x.targets))
    for client in range(clients):
        to_concat = []
        for noncontig_offsets in offsets[:, client:client + 2]:
            to_concat.append(index_source[slice(*noncontig_offsets)])
        indices.append(torch.cat(to_concat))
        n_classes_by_client.append(sum(1 for x in to_concat if x.numel() > 0))

    n_indices = np.array([x.numel() for x in indices])
    
    return indices, n_indices, n_classes_by_client


def distribute_clients_dirichlet_original(train, test, clients=400, beta=0.1, rng=None):
    '''Distribute a dataset according to a Dirichlet distribution.
    '''

    rng = np.random.default_rng(rng)

    unique = torch.Tensor(train.targets).unique()    #去掉重复的

    # Generate Dirichlet samples
    alpha = np.ones(clients) * beta     #(400)



    p = rng.dirichlet(alpha, size=len(unique))   #(10,400)


    # Get indices for train and test sets
    train_idx, _, __ = distribute_clients_categorical(train, p, clients=clients, beta=beta)
    test_idx, _, __ = distribute_clients_categorical(test, p, clients=clients, beta=beta)



    return train_idx, test_idx


def hetero_dir_partition(set, num_clients, beta, min_require_size=None):
    """

    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \\text{Dir}_{J}({\\alpha})` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.

    Sample number for each client is decided in this function.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    a=np.random.randint(100000)

    np.random.seed(seed=123)

    num_classes=len(torch.Tensor(set.targets).unique())
    
    if min_require_size is None:
        min_require_size = num_classes
    targets=set.targets
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]


    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(beta, num_clients))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in  #先算除
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(num_clients):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])

    np.random.seed(a)
    return client_dict

def client_inner_dirichlet_partition(set, num_clients, num_classes, beta, verbose=True):

    a=np.random.randint(100000)
    np.random.seed(seed=123)
    targets=set.targets
    print('type(targets)',type(targets))
    print('len(targets)',len(targets))
    # client_sample_nums=[500]*100
    client_sample_nums=[int(len(targets)/num_clients)]*num_clients
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(alpha=[beta] * num_classes,    #num_clients*num_classes
                                       size=num_clients)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]   #[[],[],[],[]]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]  

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]
    prior_list=[]
    for a in range(num_clients):
        prior_list.append([(i,class_priors[a][i]) for i in range(num_classes)])


    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        # if verbose:
            # print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        while True:
            curr_prior=[value for key,value in prior_list[curr_cid]]
            prior_cumsum = np.cumsum(curr_prior)
            curr_idx = np.argmax(np.random.uniform(0,prior_cumsum[-1]) <= prior_cumsum)
            curr_class=prior_list[curr_cid][curr_idx][0]

            if class_amount[curr_class] <= 0:
                for index,(key, value) in enumerate(prior_list[curr_cid]):
                    if key==curr_class:
                        prior_list[curr_cid].pop(index)
                print('这个类已经分完了')
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    
    np.random.seed(a)
    return client_dict


def distribute_iid(train, test, clients=200, rng=None):
    '''Distribute a dataset in an iid fashion, i.e. shuffle the data and then
    partition it.'''
    a=np.random.randint(100000)

    np.random.seed(seed=123)

    rng = np.random.default_rng(rng)

    train_idx = np.arange(len(train.targets))
    rng.shuffle(train_idx)
    train_idx = train_idx.reshape((clients, int(len(train_idx) / clients)))

    test_idx = np.arange(len(test.targets))
    rng.shuffle(test_idx)
    test_idx = test_idx.reshape((clients, int(len(test_idx) / clients)))
    np.random.seed(seed=a)

    return train_idx, test_idx


def get_mnist_or_cifar10(dataset='mnist', clients=200,mode='dirichlet', beta=0.1, batch_size=32,rng=None):
    

    if dataset not in ('mnist', 'cifar10', 'cifar100'):
        raise ValueError(f'unsupported dataset {dataset}')

    path = os.path.join('/148Dataset/data-zhang.ziyang', dataset)

    rng = np.random.default_rng(rng)


    if dataset == 'mnist':
        xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
        train = torchvision.datasets.MNIST(path, train=True, download=True, transform=xfrm)
        test = torchvision.datasets.MNIST(path, train=False, download=True, transform=xfrm)
        num_classes=10

    elif dataset == 'cifar10':
        xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        train = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=xfrm)
        test = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=xfrm)
        num_classes=10

    elif dataset == 'cifar100':
        xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        train = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=xfrm)
        test = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=xfrm)
        num_classes=100

    if mode == 'dirichlet':
        train_idx, test_idx = distribute_clients_dirichlet_original(train, test, clients=clients, beta=beta)

    elif mode=='dirichlet_hetero':
        train_idx=hetero_dir_partition(train,clients,beta=beta)
        test_idx=hetero_dir_partition(test,clients,beta=beta,min_require_size=1)
    elif mode=='balanced_dirichlet':
        train_idx=client_inner_dirichlet_partition(train, clients, num_classes=num_classes, beta=beta, verbose=True)
        test_idx=client_inner_dirichlet_partition(test, clients, num_classes=num_classes, beta=beta, verbose=True)
        
    elif mode == 'iid':
        train_idx, test_idx = distribute_iid(train, test, clients=clients)

   

    # Generate DataLoaders
    loaders = {}
    split=[]
    for i in range(clients):
        train_sampler = torch.LongTensor(train_idx[i])
        split.append(len(train_idx[i]))
        
        test_sampler = torch.LongTensor(test_idx[i])

        if len(train_sampler) == 0 or len(test_sampler) == 0:
            # ignore empty clients
            continue

        # shuffle
        train_sampler = rng.choice(train_sampler, size=train_sampler.shape, replace=False)  #这一步就是打乱
        test_sampler = rng.choice(test_sampler, size=test_sampler.shape, replace=False)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                  sampler=test_sampler)
        loaders[i] = (train_loader, test_loader)
    print('split',split)
    whole_test_loader=torch.utils.data.DataLoader(test, batch_size=batch_size)
    return loaders,whole_test_loader


def get_mnist(**kwargs):
    return get_mnist_or_cifar10('mnist', **kwargs)


def get_cifar10(**kwargs):
    return get_mnist_or_cifar10('cifar10', **kwargs)


def get_cifar100(**kwargs):
    return get_mnist_or_cifar10('cifar100', **kwargs)


def get_emnist(path='../leaf/data/femnist/data', min_samples=0, batch_size=32,
               val_size=0.2, **kwargs):
    '''Read the Federated EMNIST dataset, from the LEAF benchmark.
    The number of clients, classes per client, samples per class, and
    class imbalance are all provided as part of the dataset.

    Parameters:
    path : str
        dataset root directory
    batch_size : int
        batch size to use for DataLoaders
    val_size : float
        the relative proportion of test samples each client gets

    Returns: dict of client_id -> (train_loader, test_loader)
    '''

    EMNIST_SUBDIR = 'all_data'

    loaders = {}
    for fn in tqdm(os.listdir(os.path.join(path, EMNIST_SUBDIR))):
        fn = os.path.join(path, EMNIST_SUBDIR, fn)
        with open(fn) as f:
            subset = json.load(f)

        for uid in subset['users']:
            user_data = subset['user_data'][uid]
            data_x = (torch.FloatTensor(x).reshape((1, 28, 28)) for x in user_data['x'])
            data = list(zip(data_x, user_data['y']))

            # discard clients with less than min_samples of training data
            if len(data) < min_samples:
                continue

            n_train = int(len(data) * (1 - val_size))
            data_train = data[:n_train]
            data_test = data[n_train:]
            train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

            loaders[uid] = (train_loader, test_loader)


    return loaders


def get_dataset(dataset, **kwargs):
    '''Fetch the requested dataset, caching if needed

    Parameters:
    dataset : str
        either 'mnist' or 'emnist'
    **kwargs
        passed to get_mnist or get_emnist

    Returns: dict of client_id -> (device, train_loader, test_loader)
    '''
    print('get dataset:')
    DATASET_LOADERS = {
        'mnist': get_mnist,
        'emnist': get_emnist,
        'cifar10': get_cifar10,
        'cifar100': get_cifar100
    }

    if dataset not in DATASET_LOADERS:
        raise ValueError(f'unknown dataset {dataset}. try one of {list(DATASET_LOADERS.keys())}')


    loaders,whole_test_loader= DATASET_LOADERS[dataset](**kwargs)

    new_loaders = {}
    for i, (uid, (train_loader, test_loader)) in enumerate(loaders.items()):
        train_data = [(x.cuda(), y.cuda()) for x, y in train_loader]
        test_data = [(x.cuda(), y.cuda()) for x, y in test_loader]

        new_loaders[uid] = (train_data, test_data)

    whole_test_data=[(x.cuda(), y.cuda()) for x, y in whole_test_loader]
    return new_loaders,whole_test_data

