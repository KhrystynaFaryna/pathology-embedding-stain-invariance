import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import random
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from dataset import get_dataloaders, num_class
from torch.autograd import Variable
from resnet import ResNet
from datetime import datetime
#from torchsummary import summary
import wandb
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")
 
'''
./c-submit --require-gpu-mem="8G" --gpu-count="1" --require-mem="30G" --require-cpus="8" --priority="low" khrystynafaryna 9548 72 doduo1.umcn.nl/khrystynafaryna/kf_container_invariant:latest python3 /data/pathology/projects/autoaugmentation/from_chansey_review/invariant/inference.py  --train_set val_rh_umcu_ --val_set1 test_cwh_ --val_set2 test_lpe_ --dataset camelyon17 --dataroot '/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/'

./c-submit --require-gpu-mem="8G" --gpu-count="1" --require-mem="30G" --require-cpus="8" --priority="high" khrystynafaryna 9548 72 doduo1.umcn.nl/khrystynafaryna/kf_container_invariant:latest python3 /data/pathology/projects/autoaugmentation/from_chansey_review/invariant/inference.py  --dataroot '/data/pathology/projects/autoaugmentation/from_chansey_upd/data/midog/patches/' --dataset midog --train_set training_hamamatsu_xr_ --val_set  validation_hamamatsu_s360_ 

./c-submit --require-gpu-mem="8G" --gpu-count="1" --require-mem="30G" --require-cpus="8" --priority="low" khrystynafaryna 9548 72 doduo1.umcn.nl/khrystynafaryna/kf_container_invariant:latest python3 /data/pathology/projects/autoaugmentation/from_chansey_review/invariant/inference.py  --train_set val_rh_umcu_ --val_set1 test_cwh_ --val_set2 test_lpe_ --dataset camelyon17 --dataroot '/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/'


'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/tiger/patches/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='camelyon17',choices=['camelyon17','none','tiger','midog'],
                    help='location of the data corpus')
parser.add_argument('--train_set', type=str, default='val_A2_jb_', help='train file name')
parser.add_argument('--val_set1', type=str, default='test_aperio_cs2_', help='val file name')
parser.add_argument('--val_set2', type=str, default='test_aperio_cs2_', help='val file name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.003, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.00003, help='min learning rate')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model', type=str, default='resnet18', help='path to save the model')
parser.add_argument('--save', type=str, default='/data/pathology/projects/autoaugmentation/from_chansey_review/invariant/final_results/', help='experiment name')
parser.add_argument('--load', type=str, default='/data/pathology/projects/autoaugmentation/from_chansey_review/invariant/experiments/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--sample_weighted_loss', type=bool, default=True, help="sample weights in loss function")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--use_parallel', type=bool, default=False, help="use data parallel default False")
parser.add_argument('--num_workers', type=int, default=8, help="num_workers")
parser.add_argument('--randaugment', type=bool, default=False, help='use randaugment augmentation')
parser.add_argument('--m', type=int, default=0, help="magnitude of randaugment")
parser.add_argument('--n', type=int, default=0, help="number of layers randaugment")
parser.add_argument('--k', type=int, default=16, help="number of group elements")
parser.add_argument('--coef', type=int, default=30, help="range of a transforms, k/coef <1")
parser.add_argument('--randomize', type=bool, default=False, help="randomize magnitude in randaugment")
parser.add_argument('--randaugment_transforms_set', type=str, default='invariant_loop',choices=['review','midl2021','original','midl2021_tr2eb','midl2021_trsh2eb','invariant','manual',"invariant_loop"],help='which set of randaugment transforms to use')
parser.add_argument('--lr_schedule', type=str, default='rlop', choices = ['cos','exp','rlop'], help = "which lr scheduler to use")
parser.add_argument('--optimizer_type', type=str, default='adam', choices = ['sdg','adam','rms'], help = "which optimizer to use")
parser.add_argument('--save_best', type=bool, default=True, help="If True, updating model weights only whe minimum of loss occurs, else updating weights every epoch")

args = parser.parse_args()
arg_dict={}

arg_dict['val_set']=args.train_set
arg_dict['batch_size']=args.batch_size
arg_dict['learning_rate']=args.learning_rate
arg_dict['learning_rate_min']=args.learning_rate_min
arg_dict['randaugment']=args.randaugment
arg_dict['randomize']=args.randomize
arg_dict['m']=args.m
arg_dict['n']=args.n
arg_dict['k']=args.k
arg_dict['coef']=args.coef
arg_dict['randaugment_transforms_set']=args.randaugment_transforms_set
arg_dict['optimizer_type']=args.optimizer_type
arg_dict['lr_schedule']=args.lr_schedule
arg_dict['dataset']=args.dataset

project_name='randaugment_validation'+args.dataset+args.train_set[3:-1]
m_range=[0]#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
n_range=[0]#[1,2,3,4,5,6,7]
if args.dataset =='camelyon17':
    test_list=None
elif args.dataset == "tiger":
    test_list = ['HN','C8','A1', 'A2','A7','A8','AC','AN','AO','AQ','AR','BH','D8','E2','E9','EW','GI','GM','LL','OL','S3','jb']
elif args.dataset == "midog":
    test_list = ['test_aperio_cs2','validation_hamamatsu_s360']
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
sub_policies=None
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
LogSoftmax = torch.nn.LogSoftmax()
num_classes=2


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  model = ResNet(dataset='2d-nxk', depth=18, num_classes=num_classes, bottleneck=True, k=args.k)
  print(model)
  model = model.cuda()
  
  #summary(model,(3,128,128))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()




 
  

  


  
  if test_list is not None: 
    for test_set in test_list:
          print("prcessing dataset:",test_set)

          #load data 1
          train_queue, test1_queue = get_dataloaders(args.dataset, args.batch_size, args.num_workers, dataroot=args.dataroot, train_set=args.val_set1, val_set=test_set+"_", k=args.k, coef=args.coef)
         

         
          for n in n_range:
              for m in m_range:
                #load model
                
                data_m_n_dir = os.path.join(args.load,args.dataset+'_'+args.train_set+'n_'+str(n)+'_m_'+str(m)+'_t_'+args.randaugment_transforms_set+"_k_"+str(args.k)+"_coef_"+str(args.coef))
                
                print("dir:",data_m_n_dir) 
                if os.path.exists(data_m_n_dir):
                  time_stamp = os.listdir(data_m_n_dir)
                  print("Directory exists, it has",time_stamp) 
                  if len(time_stamp)==1:
                      print('accessing element:',time_stamp[0])
                      destination_folder = os.path.join(data_m_n_dir,time_stamp[0])
                      destination_folder_list = os.listdir(destination_folder)
                      if 'weights.pt' in destination_folder_list:

                          model_path=os.path.join(destination_folder,'weights.pt')
                          utils.load(model, model_path)
                          valid1_auc, valid1_f1, valid1_obj = infer(test1_queue, model, criterion, args.save+args.train_set+time_stamp[0]+'.csv',test_set)
                          print("auc:",valid1_auc)
                          print("f1:",valid1_f1)
                          utils.log_to_file(io_path = args.save+"/"+args.train_set+"log.csv", test_set = test_set, n=n, m=m,  valid_obj = valid1_obj, valid_metric = valid1_auc)
                      
                      else:
                        print("Subfolder has no weights.pt file")
                  else:
                
                    for d in time_stamp:
                      print('d',d)
                      destination_folder = os.path.join(data_m_n_dir,d)
                      destination_folder_list = os.listdir(destination_folder)


                      if 'weights.pt' in destination_folder_list:

                          model_path=os.path.join(destination_folder,'weights.pt')
                          utils.load(model, model_path)
                          valid1_auc, valid1_f1, valid1_obj = infer(test1_queue, model, criterion, args.save+args.train_set+d+'.csv',test_set)
                          print("auc:",valid1_auc)
                          print("f1:",valid1_f1)
                          utils.log_to_file(io_path = args.save+"/"+args.train_set+"log.csv", test_set = test_set, n=n, m=m,  valid_obj = valid1_obj, valid_metric = valid1_auc)
                      
                      else:
                        print("Subfolder has no weights.pt file")
                else:
                  print("Directory does not exist")
                  
          del train_queue, test1_queue


  else:
      
      #load data 1
      train_queue, test1_queue = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        dataroot=args.dataroot, train_set=args.train_set, val_set=args.val_set1, k=args.k, coef=args.coef)
     

     
      for n in n_range:
          for m in m_range:
            #load model
            data_m_n_dir = os.path.join(args.load,args.dataset+'_'+args.train_set+'n_'+str(n)+'_m_'+str(m)+'_t_'+args.randaugment_transforms_set+"_k_"+str(args.k)+"_coef_"+str(args.coef))
            
            print("dir:",data_m_n_dir) 
            if os.path.exists(data_m_n_dir):
              time_stamp = os.listdir(data_m_n_dir)
              print("Directory exists, it has",time_stamp) 
              if len(time_stamp)==1:
                  print('accessing element:',time_stamp[0])
                  destination_folder = os.path.join(data_m_n_dir,time_stamp[0])
                  destination_folder_list = os.listdir(destination_folder)
                  if 'weights.pt' in destination_folder_list:

                      model_path=os.path.join(destination_folder,'weights.pt')
                      utils.load(model, model_path)
                      valid1_auc, valid1_f1, valid1_obj = infer(test1_queue, model, criterion,args.save+args.train_set+time_stamp[0]+'.csv',args.val_set1)
                      print("auc:",valid1_auc)
                      print("f1:",valid1_f1)
                      utils.log_to_file(io_path = args.save+"/"+args.train_set+"log.csv", test_set = args.val_set1, n=n, m=m,  valid_obj = valid1_obj, valid_metric = valid1_auc)
                  
                  else:
                    print("Subfolder has no weights.pt file")
              else:
            
                for d in time_stamp:
                  print('d',d)
                  destination_folder = os.path.join(data_m_n_dir,d)
                  destination_folder_list = os.listdir(destination_folder)


                  if 'weights.pt' in destination_folder_list:

                      model_path=os.path.join(destination_folder,'weights.pt')
                      utils.load(model, model_path)
                      valid1_auc, valid1_f1, valid1_obj = infer(test1_queue, model, criterion,args.save+args.train_set+d+'.csv',args.val_set1)
                      print("auc:",valid1_auc)
                      print("f1:",valid1_f1)
                      utils.log_to_file(io_path = args.save+"/"+args.train_set+"log.csv", test_set = args.val_set1, n=n, m=m,  valid_obj = valid1_obj, valid_metric = valid1_auc)
                  
                  else:
                    print("Subfolder has no weights.pt file")
            else:
              print("Directory does not exist")
              
      del train_queue, test1_queue

      train_queue, test2_queue = get_dataloaders(
            args.dataset, args.batch_size, args.num_workers,
            dataroot=args.dataroot, train_set=args.train_set, val_set=args.val_set2, k=args.k, coef=args.coef)


      for n in n_range:
          for m in m_range:
              #load model
            data_m_n_dir = os.path.join(args.load,args.dataset+'_'+args.train_set+'n_'+str(n)+'_m_'+str(m)+'_t_'+args.randaugment_transforms_set+"_k_"+str(args.k)+"_coef_"+str(args.coef))
            
            print("dir:",data_m_n_dir)
            if os.path.exists(data_m_n_dir):
                time_stamp = os.listdir(data_m_n_dir) 
                if len(time_stamp)==1:
                  destination_folder = os.path.join(data_m_n_dir,time_stamp[0])
                  destination_folder_list = os.listdir(destination_folder)
                  if 'weights.pt' in destination_folder_list:

                      model_path=os.path.join(destination_folder,'weights.pt')
                      utils.load(model, model_path)
                      valid2_auc, valid2_f1, valid2_obj = infer(test2_queue, model, criterion,args.save+args.train_set+time_stamp[0]+'.csv',args.val_set2)
                      print("auc:",valid2_auc)
                      print("f1:",valid2_f1)
                      utils.log_to_file(io_path = args.save+"/"+args.train_set+"log.csv", test_set = args.val_set2, n=n, m=m,  valid_obj = valid2_obj, valid_metric = valid2_auc)
                  
                  else:
                    print("Subfolder has no weights.pt file")
                else:
                  for d in time_stamp:
                    destination_folder = os.path.join(data_m_n_dir,d)
                    destination_folder_list = os.listdir(destination_folder)


                    if 'weights.pt' in destination_folder_list:

                      model_path=os.path.join(destination_folder,'weights.pt')
                      utils.load(model, model_path)
                      valid2_auc, valid2_f1, valid2_obj = infer(test2_queue, model, criterion,args.save+args.train_set+d+'.csv',args.val_set2)
                      print("auc:",valid2_auc)
                      print("f1:",valid2_f1)
                      utils.log_to_file(io_path = args.save+args.train_set+"log.csv", test_set = args.val_set2, n=n, m=m, valid_obj = valid2_obj, valid_metric = valid2_auc)
                    else:
                      pass
            else:
              print("Directory does not exist")
      del train_queue, test2_queue



def infer(valid_queue, model, criterion, save_file_name,dataset):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    log_soft = nn.Softmax()#nn.LogSoftmax()
    target_ = []
    pred_ = []
    
    model.eval()

    pred_bin_r = []

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):

            input = Variable(input.type(torch.FloatTensor)).cuda()
            target = Variable(target).cuda(non_blocking=True)


            logits = model(input)

            loss = criterion(logits, target)
            #print('logits',torch.sigmoid(logits).detach().cpu().numpy())
            batch_weight = Variable(torch.Tensor(compute_label_weights(target.detach().cpu().numpy())), requires_grad=False).cuda()
            loss = loss*batch_weight
            loss = torch.mean(loss)
            target_.append(target.detach().cpu().numpy())
            pred_.append(log_soft(logits).detach().cpu().numpy()[:,1])
            pred_bin = log_soft(logits).detach().cpu().numpy()[:,1]
            #print('pred',pred_bin)
            pred_bin[pred_bin>0.5]=1
            pred_bin[pred_bin<=0.5]=0
            prec1 = accuracy_score(target.detach().cpu().numpy(),pred_bin)  

            
            n = input.size(0)

            objs.update(loss.detach().cpu().numpy(), n)
            top1.update(prec1, n)
            pred_bin_r.append(pred_bin)


            #if step % args.report_freq == 0:
            #    logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
    target_ = np.concatenate((target_), axis = 0)
    pred_ = np.concatenate((pred_), axis = 0)
    pred_bin_r = np.concatenate((pred_bin_r), axis = 0)
    utils.log_sample_pred_label_to_file_a(save_file_name, pred_, target_, dataset)
    auc = roc_auc_score(target_,pred_)  

    f1 = f1_score(target_,pred_bin_r)      
    return auc, f1, objs.avg




def compute_label_weights(y_true, one_hot=False):

    if one_hot:
        y_true_single = np.argmax(y_true, axis=-1)
    else:
        y_true_single = y_true

    w = np.ones(y_true_single.shape[0])
    for idx, i in enumerate(np.bincount(y_true_single)):
        w[y_true_single == idx] *= 1/(i / float(y_true_single.shape[0]))

    return w


if __name__ == '__main__':
    print('Entering main...')
    main()



