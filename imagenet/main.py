import os
import torch
import torch.distributed as dist
#from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.multiprocessing import Process

from torchvision import transforms
import sys
from resnet import resnet50
from datetime import datetime
import numpy as np
from data_partitioner import DataPartitioner
from parameters import para
import copy
import time

from sklearn import metrics
import pandas as pd
import random
#from loader import ImageNet
random_seed_id = 1234
torch.manual_seed(random_seed_id)
torch.manual_seed(random_seed_id)
torch.cuda.manual_seed(random_seed_id)
np.random.seed(random_seed_id)
random.seed(random_seed_id)
torch.backends.cudnn.deterministic=True


root = "/Users/*****/data/imagenet/"
# Model Averaging
def average_model(model, group):
    size = float(dist.get_world_size())
    for param in model.parameters():
        # all reduce
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size

def average_all(model, a, b, alpha, global_total_pos, global_total_neg, local_total_pos, local_total_neg, group):
    average_model(model, group)
    
    dist.all_reduce(a.data, op=dist.ReduceOp.SUM)
    dist.all_reduce(b.data, op=dist.ReduceOp.SUM)
    dist.all_reduce(alpha.data, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_total_pos, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_total_neg, op=dist.ReduceOp.SUM)
    
    global_total_pos += local_total_pos
    global_total_neg += local_total_neg

    a.data /= float(size)
    b.data /= float(size)
    alpha.data /= float(size)

def dppd_sg(model, a, b, alpha, model0, a0, b0, alpha0, lr, gamma):
    # Primal
    a.data = a.data - lr*(a.grad.data + 1/gamma*(a.data - a0.data))
    b.data = b.data - lr*(b.grad.data + 1/gamma*(a.data - a0.data))
    for name, param in model.named_parameters():
        param.data = param.data - lr*(param.grad.data + 1/gamma*(param.data - model0[name]))
   
    # Dual
    alpha = alpha + lr*alpha.grad.data
    

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.x = inputs
        self.y = targets

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
         return (self.x[idx], self.y[idx])
     
        
def AUC(label, scores):
    fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=1)
    return metrics.auc(fpr, tpr)

def train(rank, size, group):
    torch.cuda.set_device(para.local_rank)
    use_average = False;
    p_pos_class = (para.split_index + 1.0)/1000
    p = p_pos_class / (p_pos_class + (1-p_pos_class)*para.neg_keep_ratio) 
    configs = '_size_' + str(size) + '_lr_' + str(para.lr) + '_T0_' + str(para.T0)\
              + '_gamma_' + str(para.gamma) + '_p_%.2f'%(p) + '_I_' + str(para.I)\
              + '_local_batchsize_' + str(para.local_batchsize) + '_ImageNet_ResNet50_128'
    if 0 == rank:
        print("configs: " + configs)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Divide the data assume each machine has access to all data
    dataset = torchvision.datasets.ImageNet(root, split='train',
              transform=transform, download=None)
    print("data set ready")
     
    partition_size = [para.test_ratio] + [(1-para.test_ratio) / size  for _ in range(size)]
    partition = DataPartitioner(dataset, partition_size, seed=123, neg_keep_ratio=para.neg_keep_ratio)
    test_partition = partition.use(0) # use the first partition to test
    train_partition = partition.use(dist.get_rank() + 1)
    
    print("partition done")
    # print("len(train_data): " + str(train_partition.dataset))
    
    train_loader = torch.utils.data.DataLoader(train_partition, batch_size=para.local_batchsize, 
                                               shuffle=True, num_workers=4)
    if rank==0:
        test_loader = torch.utils.data.DataLoader(test_partition, batch_size=para.test_batchsize,
                                             shuffle=False, num_workers=4)
    
    time_spent_on_training = 0
    time_spent_list = list()
    iteration_list  =list()
    net = resnet50(pretrained=True)
    net = net.cuda()
    a = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)
    alpha = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True)

    t_total = 0
    local_total_pos = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
    local_total_neg = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
    global_total_pos = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
    global_total_neg = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
    p_hat = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
    auc_history = list()

    end_all = False
    net.zero_grad()
    start_eval_time = time.time()
    
    print("doing average before entering training")
    with torch.no_grad():
        average_all(net, a, b, alpha, global_total_pos, global_total_neg, local_total_pos, local_total_neg, group) 
    print("Start Training")
    for s in np.arange(1, para.numStages):
        torch.cuda.empty_cache()
        if True == end_all:
            break
        if s > 1:
            for name, param in net.named_parameters():
                param.data = net_average[name]
            a.data = a_average
            b.data = b_average
        
        net0 = copy.deepcopy(net.state_dict())
        
        a0 = a.clone().detach().requires_grad_(False)
        b0 = b.clone().detach().requires_grad_(False)
        alpha0 = alpha.clone().detach().requires_grad_(False)

        # To take and keep average at each communication
        net_average = copy.deepcopy(net.state_dict())
        a_average = a0.clone().detach()
        b_average = b0.clone().detach()

        # evaluate alpha for current w_0
        h_neg = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
        N_neg = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
        h_pos = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
        N_pos = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False)
        net.eval()
        with torch.no_grad():
            for _ in range(3**(s)):
               try:
                   tmp_data, tmp_label = train_iter.next()
               except:
                   print("Contructing a new train_loader")
                   train_iter = iter(train_loader)
                   tmp_data, tmp_label = train_iter.next()
           
               tmp_data = tmp_data.cuda()
               tmp_label = tmp_label.cuda()
               tmp_label[tmp_label <= para.split_index] = -1
               tmp_label[tmp_label > para.split_index] = 1

               h_neg += torch.sum(net(tmp_data)[:, 1]*(-1==tmp_label).float())
               N_neg += torch.sum(-1 == tmp_label)
               h_pos += torch.sum(net(tmp_data)[:, 1]*(1==tmp_label).float())
               N_pos += torch.sum(1 == tmp_label)

        net.train()

        dist.all_reduce(h_neg, op=dist.ReduceOp.SUM)
        dist.all_reduce(N_neg, op=dist.ReduceOp.SUM)
        dist.all_reduce(h_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(N_pos, op=dist.ReduceOp.SUM)
       
        alpha.data = h_neg/N_neg - h_pos/N_pos

        a0 = a.clone().detach().requires_grad_(False)
        b0 = b.clone().detach().requires_grad_(False)
        alpha0 = alpha.clone().detach().requires_grad_(False)
        T = para.T0*(3**(s-1))
        lr = para.lr*((1/3)**(s-1))
        
        # To take and keep average at each communication
        net_average = copy.deepcopy(net.state_dict())
        a_average = a0.clone().detach()
        b_average = b0.clone().detach()
        
        for t in range(T):
            if (0 == t%100) & (rank == 0):
                print("t: " + str(t))
 
            # start evaluation
            if 0 == t_total % para.test_freq:
               eval_start_time = time.time()
               net.eval()
               
               ## average the average dictionary
               if use_average ==True:
                 # use the average model to evalute, but keep the current model
                 tmp_model = copy.deepcopy(net.state_dict())
                 model_eval = copy.deepcopy(net_average)
               
                 for name, param in net.named_parameters():
                    dist.all_reduce(model_eval[name], op=dist.ReduceOp.SUM)
                    model_eval[name] /= size
                    if 0 == rank:
                        param.data = model_eval[name]
                   #average_model(net, group)

               if 0 == rank:
                   print("starting eval")
                   score_list = list()
                   label_list = list()

                   for _, data in enumerate(test_loader, 0):
                       tmp_data, tmp_label = data
                       tmp_data = tmp_data.cuda()
                       # tmp_label = tmp_label.cuda()
                       tmp_label[tmp_label <= para.split_index] = -1
                       tmp_label[tmp_label > para.split_index] = 1
                       tmp_score = net(tmp_data)[:, 1].detach().clone().cpu()
                       score_list.append(tmp_score)
                       label_list.append(tmp_label)
                   # print("data: " + str(tmp_data))
                   test_label = torch.cat(label_list)
                   test_score = torch.cat(score_list)
                   # print("test_score: "  + str(test_score)) 
                   auc = AUC(test_label, test_score)
                   auc_history.append(auc)
                   print(datetime.now(),"Stage: " + str(s) + "; Iter: " + str(t_total)\
                         +'; lr: %.3f '%(lr) + "; p_hat: " + str(p_hat.cpu().numpy()[0])\
                         + "; auc: %.4f"%(auc))
            
                   tmp_time = time_spent_on_training
                   time_spent_list.append(tmp_time)
                   iteration_list.append(t_total)
                   df = pd.DataFrame(data={'total_iteration':iteration_list, \
                                     'time':time_spent_list, 'Test'+configs:auc_history})
                   df.to_csv('history/history'+configs+'.csv')
                   del tmp_score
                   # after evaluation, assign back the model in trajectory
                   if use_average == True:
                     for name, param in net.named_parameters():
                         param.data = tmp_model[name]
                   eval_duration = time.time() - eval_start_time
                   start_eval_time -= eval_duration

               net.train()
	    # end of evaluation 

            if t_total > para.total_iter:
                end_all = True
                break
 
            try:
                tmp_data, tmp_label = train_iter.next()
            except:
                print("Contructing a new train_loader")
                train_iter = iter(train_loader)
                tmp_data, tmp_label = train_iter.next()
                
 
            tmp_data = tmp_data.cuda()
            tmp_label = tmp_label.cuda()

            start_iteration = datetime.now()
            t_total += 1

            # communicate over all nodes 
            if 0 == t_total%(para.I):
                with torch.no_grad():
                    if size > 1:
                        average_all(net, a, b, alpha, global_total_pos, \
                                    global_total_neg, local_total_pos, local_total_neg, group) 
                    else:
                        global_total_neg += local_total_neg
                        global_total_pos += local_total_pos
                    local_total_neg = 0
                    local_total_pos = 0
  
            tmp_label[tmp_label <= para.split_index] = -1
            tmp_label[tmp_label > para.split_index] = 1
            

            local_total_pos += torch.sum(1 == tmp_label)
            local_total_neg += torch.sum(-1 == tmp_label)
            p_hat = p_hat*0 + float(global_total_pos + local_total_pos)/\
                              float(global_total_pos + local_total_pos + global_total_neg + local_total_neg)
            tmp_score = net(tmp_data)[:, 1]
            
            loss = (1-p_hat)*torch.mean((tmp_score - a)**2*(1==tmp_label).float())\
                   + p_hat*torch.mean((tmp_score - b)**2*(-1==tmp_label).float())\
                   + 2*(1+alpha) *torch.mean((p_hat*tmp_score*(-1==tmp_label).float()\
                                             - (1-p_hat)*tmp_score*(1==tmp_label).float()))\
                   - p_hat*(1-p_hat)*(alpha**2)
            net.zero_grad()
            try:
                a.grad.data *= 0
                b.grad.data *= 0
                alpha.grad *= 0 
            except:
                pass
         
            loss.backward(retain_graph=True)
            dppd_sg(net, a, b, alpha, net0, a0, b0, alpha0, lr, para.gamma)
            del loss
            del tmp_score

            end_iteration = datetime.now()
   
            for name, param in net.named_parameters():
                net_average[name] = net_average[name] + param.data
                
            time_spent_on_training += (end_iteration - start_iteration).total_seconds() 
        # average on T
        for name, param in net.named_parameters():
            net_average[name] = net_average[name]/T
    

if __name__ == "__main__":
    #size = para.numGPU
    os.environ['MASTER_ADDR'] = para.master_addr 
    dist.init_process_group('nccl')
    size = dist.get_world_size()
    group = dist.new_group(range(size))
    rank = dist.get_rank()
    print("intilized, rank: " + str(rank) + "size: " + str(size))
    train(rank, size, group)


