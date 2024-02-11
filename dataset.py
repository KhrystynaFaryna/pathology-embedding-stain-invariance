import torch
import torchvision
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from augmenters.color.hedcoloraugmenter import HedColorAugmenter
import random
import string
import matplotlib.pyplot as plt

class Camelyon17(Dataset):
    """Custom dataset for reading a datasets of .npy arrays with patches."""

    def __init__(self, x_path, y_path, transform=None, augmenter=None, k =16,coef=30,dataset='camelyon17'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_path = x_path
        self.y_path = y_path
        self.k=k
        self.coef = coef
        self.augmenter = augmenter
        self.x = np.load(self.x_path)
        self.targets = np.load(self.y_path)
 
        self.dataset = dataset


        if self.dataset=="midog":

            # midog modifications
            self.targets[self.targets=="hard negative" ]=0
            self.targets[self.targets=="mitotic figure" ]=1
            self.targets[self.targets=="negative generated"]=0
     
            print('modified to ', len(np.unique(self.targets)),'classes')


        print('In the dataloader. Loaded the images.')

    def __len__(self):
        print('Length of the dataset:',len(self.targets))
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.patches= (self.x[idx, ...])
        self.y=self.targets[idx]
        # printing lowercase
        letters = string.ascii_lowercase
        im_name = ''.join(random.choice(letters) for i in range(10)) 
        cyclic_shift_range=range(self.k)
        cyclic_shift = random.choice(cyclic_shift_range)

        self.x_stack = np.zeros((self.patches.shape[0],self.patches.shape[1],self.patches.shape[2],self.k))
        for i in range(self.k):

            temp_im = hed_local(self.patches,i/self.coef)
            self.x_stack[:,:,:,i]=temp_im

            #plt.imsave("/data/pathology/projects/autoaugmentation/from_chansey_review/invariant/imgs_inv/"+str(i)+im_name+".png",temp_im)
        self.x_stack =np.roll(self.x_stack, cyclic_shift, axis=-1)
        self.x_stack = np.moveaxis(self.x_stack, 2,0).astype(np.float32)/255.0
        return self.x_stack, self.y.astype(np.long)

def hed_local(image, factor):
    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor

    augmentor= HedColorAugmenter(haematoxylin_sigma_range=factor, haematoxylin_bias_range=factor,
                                            eosin_sigma_range=factor/10, eosin_bias_range=factor/10,
                                            dab_sigma_range=factor, dab_bias_range=factor,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hed_e'+str(num)+'.jpg')

    '''
    #print('_hed_e',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])



def num_class(dataset):
    return {
        'cifar10': 10,
        'camelyon17': 2,
        'tiger': 2,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]

def get_dataloaders(dataset, batch, num_workers, dataroot, train_set, val_set, augmenter='none',k=16,coef=30):
    

    if dataset == 'camelyon17':
        #Load train data
        x_path=dataroot+train_set+'x.npy'
        y_path=dataroot+train_set+'y.npy'

        val_x_path = dataroot+val_set+'x.npy'
        val_y_path = dataroot+val_set+'y.npy'
        #Load train data
        total_trainset = Camelyon17(x_path, y_path, transform=None,augmenter = None, k=k, coef=coef) # 4000 trainset
        total_valset = Camelyon17(val_x_path, val_y_path, transform=None, augmenter = None, k = k, coef=coef)

    elif dataset == 'midog':
        #Load train data
        x_path=dataroot+train_set+'x.npy'
        y_path=dataroot+train_set+'y.npy'

        val_x_path = dataroot+val_set+'x.npy'
        val_y_path = dataroot+val_set+'y.npy'
        #Load train data
        total_trainset = Camelyon17(x_path, y_path, transform=None,augmenter = None,k=k,coef=coef,dataset='midog') # 4000 trainset
        total_valset = Camelyon17(val_x_path, val_y_path, transform=None, augmenter = None,dataset='midog', k = k, coef=coef)

    
    elif dataset == 'mitosis':
        #Load train data
        x_path=dataroot+train_set+'x.npy'
        y_path=dataroot+train_set+'y.npy'

        val_x_path = dataroot+val_set+'x.npy'
        val_y_path = dataroot+val_set+'y.npy'
        #Load train data
        total_trainset = Camelyon17(x_path, y_path, transform=None,augmenter = None,k=k,coef=coef) # 4000 trainset
        total_valset = Camelyon17(val_x_path, val_y_path, transform=None, augmenter = None, k = k, coef=coef)

    else:
        raise ValueError('invalid dataset name=%s' % dataset)


    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(
        total_valset, batch_size=batch, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=num_workers)

    return trainloader, validloader
