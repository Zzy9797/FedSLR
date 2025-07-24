import sys
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
import os


def initialize_mask(model, dtype=torch.bool):
    layers_to_prune = (layer for _, layer in model.named_children())
    for layer in layers_to_prune:
        for name, param in layer.named_parameters():

            if name.endswith('weight'):
                if hasattr(layer, name + '_mask'):
                    warnings.warn(
                        'Parameter has a pruning mask already. '
                        'Reinitialize to an all-one mask.'
                    )
                layer.register_buffer(name + '_mask', torch.ones_like(param, dtype=dtype))
                continue


class PrunableNet(nn.Module):

    def __init__(self):
        super(PrunableNet, self).__init__()


    def init_param_sizes(self):

        self.mask_size = 0
        self.param_size = 0
        for _, layer in self.named_children():
            for name, param in layer.named_parameters():
                param_size = np.prod(param.size())
                self.param_size += param_size * 32 
                if name.endswith('weight'):
                    self.mask_size += param_size
        print(f'Masks require {self.mask_size} bits.')  
        print(f'Parameters require {self.param_size} bits.')  
        print(f'Unmasked Parameters require {self.param_size - self.mask_size*32} bits.') 



    def clear_gradients(self):
        for _, layer in self.named_children():
            for _, param in layer.named_parameters():
                del param.grad
        torch.cuda.empty_cache()
        


    def infer_mask(self, masking):
        for name, param in self.state_dict().items():
            if name.endswith('weight') and name in masking.masks:
                mask_name = name + "_mask"
                mask = self.state_dict()[mask_name]
                mask.copy_(masking.masks[name])


    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def _decay(self, t, alpha=0.6, t_end=400):
        if t >= t_end:
            return 0

        return alpha/2 * (np.exp(-t))


    def _weights_by_layer(self, sparsity=0.1, sparsity_distribution='erk'):
        with torch.no_grad():
            layer_names = []
            sparsities = np.empty(len(list(self.named_children())))
            n_weights = np.zeros_like(sparsities, dtype=np.int)

            for i, (name, layer) in enumerate(self.named_children()):

                layer_names.append(name)
                for pname, param in layer.named_parameters():
                    n_weights[i] += param.numel()
                if sparsity_distribution == 'uniform':
                    sparsities[i] = sparsity
                    continue
                
                kernel_size = None
                if isinstance(layer, nn.modules.conv._ConvNd):
                    neur_out = layer.out_channels
                    neur_in = layer.in_channels
                    kernel_size = layer.kernel_size
                elif isinstance(layer, nn.Linear):
                    neur_out = layer.out_features
                    neur_in = layer.in_features
                else:
                    raise ValueError('Unsupported layer type ' + type(layer))

                if sparsity_distribution == 'er':
                    sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                elif sparsity_distribution == 'erk':
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        sparsities[i] = 1 - (neur_in + neur_out + np.sum(kernel_size)) / (neur_in * neur_out * np.prod(kernel_size))
                    else:
                        sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                else:
                    raise ValueError('Unsupported sparsity distribution ' + sparsity_distribution)
    


            sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
        
            keep_num= np.floor((1-sparsities) * n_weights) 

        
            prune_num=n_weights-keep_num 
         

            return {layer_names[i]: keep_num[i] for i in range(len(layer_names))},{layer_names[i]: prune_num[i] for i in range(len(layer_names))} 

    def layer_prune(self, sparsity=0.1, sparsity_distribution='erk',log_path=None,server_round=0):

        with torch.no_grad():
            keep_num,prune_num = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)   
           
            
            for name, layer in self.named_children():
                
                n_prune = int(prune_num[name])  


                for pname, param in layer.named_parameters():
                    
                    if not pname.endswith('weight'):
                        
                        continue
                    
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            
                            index=torch.nonzero(buf.view(buf.numel())).squeeze()
                    if (server_round-1) %100 ==0 and log_path is not None:
                        sorted,idx=torch.sort(torch.abs(param.data.flatten()),descending=True)
                        with open(os.path.join(log_path,'round'+str(server_round), 'weight.log'), 'a') as f:
                            for a in sorted.tolist():
                                f.write(str(a)+',')
                      
                            f.write('\n')

                    b=torch.clone(torch.abs(param.data.flatten()))
                    b[index]*=1000
                    b[index]+=1

                    _, prune_indices = torch.topk(b,n_prune, largest=False)
                    
                    
                    param.data.view(param.data.numel())[prune_indices] = 0
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            
                            buf.view(buf.numel())[prune_indices] = 0
                
            


    

    def prune_and_grow(self, prune_sparsity=0.1, grow_sparsity=0.1,sparsity_distribution='erk'):
        
        with torch.no_grad():

            keep_num_prune,prune_num_prune = self._weights_by_layer(sparsity=prune_sparsity, sparsity_distribution=sparsity_distribution)  
            keep_num_grow,prune_num_grow = self._weights_by_layer(sparsity=grow_sparsity, sparsity_distribution=sparsity_distribution)
           
            
            for name, layer in self.named_children():
                
                n_prune = int(prune_num_prune[name])
                n_grow = int(prune_num_prune[name])-int(prune_num_grow[name])  

                for pname, param in layer.named_parameters():                   
                    if not pname.endswith('weight'):  
                        continue

                    for bname, buf in layer.named_buffers():

                        if bname == pname + '_mask':
                            
                            index=torch.nonzero(buf.view(buf.numel())).squeeze()
                    a=torch.clone(torch.abs(param.data.flatten()))
                    a[index]*=1000
                    a[index]+=1

                    _, prune_indices = torch.topk(a,n_prune, largest=False)
                    
            
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':

                            buf.view(buf.numel())[prune_indices] = 0
                            keep_index=torch.nonzero(buf.view(buf.numel())).squeeze()

 
                    b1=torch.clone(torch.abs(param.grad.flatten()))
                    b1[keep_index]=-1.
                    b2=torch.clone(torch.abs(param.data.flatten()))
                    b2[keep_index]=-1.
                    alpha=0.8
                    importance=alpha*F.softmax(b1)+(1-alpha)*F.softmax(b2)
                    _, grow_indices = torch.topk(importance,
                                                 n_grow, largest=True)
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            buf.view(buf.numel())[grow_indices] = 1
                            param.data[~buf]=0

    
    def layer_prune_agg(self, sparsity=0.1, sparsity_distribution='erk',agg_mask=None):
        with torch.no_grad():
            keep_num,prune_num = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)   
           
            
            for name, layer in self.named_children():
                
                n_prune = int(prune_num[name]) 


                for pname, param in layer.named_parameters():
                    
                    
                    if not pname.endswith('weight'):
                        
                        continue
                    
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            _, prune_indices = torch.topk(agg_mask[name+'.'+pname].view(buf.numel()),n_prune, largest=False)
                            buf.view(buf.numel())[:]=1
                            buf.view(buf.numel())[prune_indices] = 0
                    

                    
                    param.data.view(param.data.numel())[prune_indices] = 0
                  
            

                    


    def reset_weights(self, global_state=None, use_global_mask=False,
                      keep_local_masked_weights=False,
                      global_communication_mask=False):

        with torch.no_grad(): 
            mask_changed = False
            local_state = self.state_dict()
            if global_state is None:
                param_source = local_state
            else:
                param_source = global_state
            if use_global_mask:
                apply_mask_source = global_state
            else:
                apply_mask_source = local_state
            if global_communication_mask:
                copy_mask_source = local_state
            else:
                copy_mask_source = apply_mask_source
            new_state = {}

            for name, param in param_source.items():
                if name.endswith('_mask'):
                    continue

                new_state[name] = local_state[name]

                mask_name = name + '_mask'
                if name.endswith('weight') and mask_name in apply_mask_source:
                    
                   
                    mask_to_apply = apply_mask_source[mask_name]
                    mask_to_copy = copy_mask_source[mask_name]


                    gpu_param = param[mask_to_apply] 

                    new_state[name][mask_to_apply] = gpu_param 
                    if mask_name in local_state:
                        new_state[mask_name] = local_state[mask_name] 
                  

                    new_state[mask_name].copy_(mask_to_copy) 
                    if not keep_local_masked_weights:
                        new_state[name][~mask_to_apply] = 0

                    if mask_name not in local_state or not torch.equal(local_state[mask_name], mask_to_copy):
                        mask_changed = True
                        with open('/feddst-main/experiment/FedSLR/logtxt/record_mnist.txt', 'a') as f:
                            content = "mask change!\n"
                            f.write(content)
                        print('mask change!')
                else:
                    new_state[name].copy_(param)

            self.load_state_dict(new_state)
        return mask_changed


    def proximal_loss(self, last_state):

        loss = torch.tensor(0.).cuda()

        state = self.state_dict()
        for i, (name, param) in enumerate(state.items()):
            if name.endswith('_mask'):
                continue
            gpu_param = last_state[name].cuda()
            loss += torch.sum(torch.square(param - gpu_param))
            if gpu_param.data_ptr != last_state[name].data_ptr:
                del gpu_param

        return loss




    def sparsity(self, buffers=None):

        if buffers is None:
            buffers = self.named_buffers()

        n_weights = np.zeros_like(np.empty(len(list(self.named_children()))), dtype=np.int)

        for i, (name, layer) in enumerate(self.named_children()):
            for pname, param in layer.named_parameters():
                    
                n_weights[i] += param.numel()


        n_ones = 0
        mask_size = 0
        for name, buf in buffers:
            if name.endswith('mask'):
                n_ones += torch.sum(buf)
                mask_size += buf.nelement()

        return (mask_size - n_ones).item() / np.sum(n_weights)
    

    def zero_num(self, buffers=None):

        if buffers is None:
            buffers = self.named_buffers()

        n_ones = 0
        mask_size = 0
        for name, buf in buffers:
            if name.endswith('mask'):
                n_ones += torch.sum(buf)
                mask_size += buf.nelement()

        return mask_size - n_ones.item()

class MNISTNet(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(MNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 10, 5) 
        self.conv2 = nn.Conv2d(10, 20, 5) 

        self.fc1 = nn.Linear(20 * 16 * 16, 50)
        self.fc2 = nn.Linear(50, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CIFAR10Net(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(CIFAR10Net, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class CIFAR100Net(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(CIFAR100Net, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 100)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    

class EMNISTNet(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(EMNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 10, 5) 
        self.conv2 = nn.Conv2d(10, 20, 5)

        self.fc1 = nn.Linear(20 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 62)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Conv2(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(Conv2, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 62)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


all_models = {
        'mnist': MNISTNet,
        'emnist': Conv2,
        'cifar10': CIFAR10Net,
        'cifar100': CIFAR100Net
}

