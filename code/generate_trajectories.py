
import torch
import random
import argparse

from train import *
from utils import *
from models.generator import *
from models.discriminator import Discriminator

from opacus.grad_sample import GradSampleModule
from opacus.validators import ModuleValidator


#First import pretrained models
#Generate random noise
#Generate samples from noise 
#Save noise
#Save samples


def main(opt):
    device = torch.device("cuda:"+opt.cuda)
    if opt.data == 'geolife':
        TOTAL_LOCS = 23768 # Length of gps file
    elif opt.data == 'porto':
        TOTAL_LOCS = 10853 # length of gps file
    else:
        TOTAL_LOCS = 0
    
    DATA_PATH = '../data'
    BATCH_SIZE = opt.batch
    SEQ_LEN = opt.length
    GENERATED_NUM = opt.generated

    # Import pretrained modules
    generator = ATGenerator(device=device,total_locations=TOTAL_LOCS,starting_sample='real', starting_dist=np.load(f'../data/{opt.data}/start.npy'),data=opt.data, make_private=(opt.epsilon>0))
    generator.load_state_dict(torch.load(DATA_PATH + f'/{opt.data}/{int(opt.epsilon)}/models/generator.pth'))
    generator.eval()
    
    # Discriminator has to be from opacus and pass everything to GPU
    generator = generator.to(device)

    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM,
                            DATA_PATH+f'/{opt.data}/{opt.epsilon}/infer/samples.data', inference=True)
    
    # Check that samples are unique
    # samples = read_data_from_file(DATA_PATH+f'/{opt.data}/{opt.epsilon}/infer/samples.data')
    # samples = np.unique(samples, axis=0)
    # print(len(samples))
    # with open(DATA_PATH+f'/{opt.data}/{opt.epsilon}/infer/samples', 'w') as fout:
    #     for sample in samples:
    #         string = ' '.join([str(s) for s in sample])
    #         fout.write('%s\n' % string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',  default="0", type=str)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--generated', default=64, type=int)
    parser.add_argument('--data', default='geolife', type=str)
    parser.add_argument('--length', default=48, type=int)
    parser.add_argument('-e','--epsilon', default=5, type=int)
    opt = parser.parse_args()
    main(opt)



