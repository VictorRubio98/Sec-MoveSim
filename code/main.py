# coding=utf-8
import pdb
import torch
import random
import argparse
import setproctitle
from torch import nn, optim

from train import *
from utils import *
from rollout import Rollout
from evaluations import IndividualEval
from gen_data import *
from models.generator import *
from models.discriminator import Discriminator
from models.gan_loss import GANLoss, distance_loss, period_loss
from data_iter import GenDataIter, NewGenIter, DisDataIter

from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.validators import ModuleValidator

def main(opt):
    # all parameters
    # assigned in argparse
    print(opt)
    make_private = (opt.epsilon != -1) # Use --epsilon -1 or -e -1 for baseline results      
    
    # fixed parameters
    SEED = 2020
    EPOCHS = 30
    BATCH_SIZE = 32
    if opt.data == 'porto':
        SEQ_LEN = 40
    elif opt.data == 'geolife':
        SEQ_LEN = 48 
    GENERATED_NUM = 10016
    
    DATA_PATH = '../data'
    EPS_PATH = (DATA_PATH + f'/{opt.data}/{opt.epsilon}/') if not make_private else DATA_PATH + f'/{opt.data}/{int(opt.epsilon)}/'
    REAL_DATA = DATA_PATH + f'/{opt.data}/real.data'
    VAL_DATA = DATA_PATH + f'/{opt.data}/val.data'
    TEST_DATA = DATA_PATH + f'/{opt.data}/test.data'
    GENE_DATA = EPS_PATH + '/gene.data'

    random.seed(SEED)
    np.random.seed(SEED)

    if opt.data == 'mobile':
        TOTAL_LOCS = 8606
        individualEval = IndividualEval(data='mobile')
    elif opt.data == 'geolife':
        TOTAL_LOCS = 23768
        individualEval = IndividualEval(data='geolife')
    elif opt.data == 'porto':
        TOTAL_LOCS = 10853
        individualEval = IndividualEval(data='porto')
    
    if opt.cuda == 'cpu':
        device = torch.device(opt.cuda)
    else:
        device = torch.device("cuda:"+opt.cuda)

    print('Pre-processing Data...', flush = True)
    if not opt.skipm:
        gen_matrix(opt.data)
        print('Matrix generated', flush=True)
    else:
        print('Skipped Matrix generation', flush = True)
    # assigned according to task
    if opt.task == 'attention':
        real_eps = 0
        d_pre_epoch = 20
        g_pre_epoch = 110
        ploss_alpha = float(opt.ploss)
        dloss_alpha = float(opt.dloss)
        if opt.data == 'geolife':
            generator = ATGenerator(device=device,total_locations=TOTAL_LOCS,starting_sample='real', starting_dist=np.load(f'../data/{opt.data}/start.npy'),data=opt.data, make_private=make_private)
        else:
            generator = ATGenerator(device=device,total_locations=TOTAL_LOCS,starting_sample='zero', data=opt.data, make_private=make_private)
        discriminator = Discriminator(total_locations=TOTAL_LOCS)
        gen_train_fixstart = True
        
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    
    if make_private: #Just a formality, can be skipped
        discriminator = ModuleValidator.fix(discriminator)
        discriminator = GradSampleModule(discriminator, force_functorch=True)

    # prepare files and datas
    logger = get_workspace_logger(opt.data, opt.epsilon)

    if opt.pretrain:
        if make_private:
            privacy_engine = PrivacyEngine(accountant='rdp')
            single_epoch_eps = (opt.epsilon/(8*EPOCHS + d_pre_epoch + g_pre_epoch)) #Evenly split between all epochs
            pre_dis_eps = single_epoch_eps * d_pre_epoch
            pre_gen_eps = single_epoch_eps * g_pre_epoch
            
        # generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM,
        #                  EPS_PATH + '/gene_epoch_init.data')

        # pretrain discriminator
        logger.info('pretrain discriminator ...')
        
        pretrain_real =  DATA_PATH + f'/{opt.data}/real.data'
        pretrain_fake =  DATA_PATH + f'/{opt.data}/dispre.data'
        dis_data_iter = DisDataIter(pretrain_real, pretrain_fake, BATCH_SIZE, SEQ_LEN)
        dis_criterion = nn.NLLLoss(reduction='sum')
        dis_optimizer = optim.SGD(discriminator.parameters(),lr=0.000001)
        dis_criterion = dis_criterion.to(device)

        if make_private: 
            discriminator, dis_optimizer, dis_data_iter = privacy_engine.make_private_with_epsilon(
                    module = discriminator,
                    optimizer = dis_optimizer,
                    data_loader = dis_data_iter,
                    max_grad_norm = 10000.0,
                    noise_generator=torch.cuda.manual_seed(SEED),
                    target_epsilon = pre_dis_eps,
                    target_delta = opt.delta,
                    epochs = d_pre_epoch
                )
            
            print(f'Pre-training discriminator with epsilon {pre_dis_eps:.2f}')

            
        generator.train(False)
        discriminator.train(True)

        pretrain_model("D", d_pre_epoch, discriminator, dis_data_iter,
                       dis_criterion, dis_optimizer, BATCH_SIZE, device=device)
        
        if make_private:
            real_eps += privacy_engine.get_epsilon(opt.delta)
        
        # pretrain generator
        
        logger.info('pretrain generator ...')
        generator.train(True)
        discriminator.train(False)
        
        if make_private: #Just a formality, can be skipped
            generator = GradSampleModule(generator, force_functorch=True, batch_first = True)
        
        
        if gen_train_fixstart:
            gen_data_iter = NewGenIter(REAL_DATA, BATCH_SIZE)
        else:
            gen_data_iter = GenDataIter(REAL_DATA, BATCH_SIZE)
        gen_criterion = nn.NLLLoss(reduction='sum')
        gen_optimizer = optim.SGD(generator.parameters(),lr=0.1)
        gen_criterion = gen_criterion.to(device)
        
        if make_private:
            privacy_engine = PrivacyEngine(accountant='rdp')
            generator, gen_optimizer, gen_data_iter = privacy_engine.make_private_with_epsilon(
                    module = generator,
                    optimizer = gen_optimizer,
                    data_loader = gen_data_iter,
                    max_grad_norm = 10000.0,
                    noise_generator=torch.cuda.manual_seed(SEED),
                    target_epsilon = pre_gen_eps,
                    target_delta = opt.delta,
                    epochs = d_pre_epoch,
                    batch_first = True
                )
            print(f'Pre-training generator with epsilon {pre_gen_eps:.2f}')
            
        pretrain_model("G", g_pre_epoch, generator, gen_data_iter,
                       gen_criterion, gen_optimizer, BATCH_SIZE, device=device)
        
        if make_private:
            real_eps += privacy_engine.get_epsilon(opt.delta)
            generator = generator._module
            print(f'Epsilon used in pretrain {real_eps}',flush=True)
            
        torch.save(generator.state_dict(), EPS_PATH + '/pretrain/generator.pth')
        torch.save(discriminator.state_dict(), EPS_PATH + 'pretrain/discriminator.pth')
        generator = generator.to(torch.device('cpu'))
        discriminator = discriminator.to(torch.device('cpu'))
        
    elif not opt.pretrain and opt.load:
        generator.load_state_dict(torch.load( EPS_PATH +'/pretrain/generator.pth' ))    
        discriminator.load_state_dict(torch.load( EPS_PATH +'/pretrain/discriminator.pth'))
        print('', flush = True)
    
    print('advtrain generator and discriminator ...', flush = True)
    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.SGD(generator.parameters(),lr=0.0001)

    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.SGD(discriminator.parameters(),lr=0.000001)
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    gen_gan_loss = gen_gan_loss.to(device)
    dis_criterion = dis_criterion.to(device)

    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)
    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, EPS_PATH +'/gene_epoch_{0}.data')
    print('Samples generated starting training', flush = True)
    if make_private:
        eps = (opt.epsilon-real_eps)/(EPOCHS*4) #This variable epsilon is used everey 2 epochs. 
        print(f'Epsilon for training each epoch {eps/2:.2f}', flush = True)

    for epoch in range(EPOCHS):
        gene_data = read_data_from_file(GENE_DATA)
        val_data = read_data_from_file(VAL_DATA)

        JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=val_data, length=SEQ_LEN)

        with open( EPS_PATH +'/logs/jsd.log', 'a') as f:
            f.write(' '.join([str(j) for j in JSDs]))
            f.write('\n')
        
        print("Current JSD for epoch %d: %f, %f, %f, %f, %f, %f" % (epoch, JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]), flush = True)
        generator.train(True)
        discriminator.train(False)
        # Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, SEQ_LEN)
            # construct the input to the genrator, add zeros before samples and
            # delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            zeros = zeros.to(device)
            inputs = torch.cat([zeros, samples.data], dim=1)[
                              :, :-1].contiguous()
            tim = torch.LongTensor([i%24 for i in range(SEQ_LEN)]).to(device)
            tim = tim.repeat(BATCH_SIZE).reshape(BATCH_SIZE, -1)
            
            targets = samples.contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = torch.Tensor(rewards)
            rewards = torch.exp(rewards.to(device)).contiguous().view((-1,))
            prob = generator.forward(inputs, tim)
            
            try:
                gloss = gen_gan_loss(prob, targets, rewards, device)
            except:
                gloss = gen_gan_loss(prob, targets, rewards, device)

            if ploss_alpha != 0.:
                p_crit = period_loss(24)
                p_crit = p_crit.to(device)
                pl = p_crit(samples.float())
                gloss += ploss_alpha * pl
            if dloss_alpha != 0.:
                d_crit = distance_loss(device=device,datasets=opt.data)
                d_crit = d_crit.to(device)
                dl = d_crit(samples.float())
                gloss += dloss_alpha * dl
            gen_gan_optm.zero_grad()
            gloss.backward()
            gen_gan_optm.step()
        
        rollout.update_params()
        generator.train(False)
        discriminator.train(True)
        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, SEQ_LEN,
                             GENERATED_NUM, GENE_DATA)
            dis_data_iter = DisDataIter(REAL_DATA, GENE_DATA, BATCH_SIZE, SEQ_LEN)
            if make_private:
                privacy_engine = PrivacyEngine(accountant='rdp')
                discriminator, dis_optimizer, dis_data_iter = privacy_engine.make_private_with_epsilon(
                    module = discriminator,
                    optimizer = dis_optimizer,
                    data_loader = dis_data_iter,
                    max_grad_norm = 10000.0,
                    noise_generator=torch.cuda.manual_seed(SEED),
                    target_epsilon = eps,
                    target_delta = opt.delta,
                    epochs=2
                )
            for _ in range(2):
                dloss = train_epoch("D", discriminator, dis_data_iter, dis_criterion, dis_optimizer, BATCH_SIZE, device=device, grad_acum=opt.acum)
            if make_private:
                real_eps += privacy_engine.get_epsilon(opt.delta)

        logger.info('Epoch [%d] Generator Loss: %f, Discriminator Loss: %f, Epsilon: %.2f' %
                    (epoch, gloss.item(), dloss, real_eps))
        with open( EPS_PATH +'/logs/loss.log', 'a') as f:
            f.write(' '.join([str(j)
                              for j in [epoch, float(gloss.item()), dloss]]))
            f.write('\n')
        if (epoch + 1) % 20 == 0:
            generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM,  EPS_PATH +'/gene_epoch_{epoch+1}.data')
    
    test_data = read_data_from_file(TEST_DATA)
    gene_data = read_data_from_file(GENE_DATA)
    JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=test_data, length=SEQ_LEN)
    print("Test JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]), flush=True)
    if make_private:
        print(f'Final epsilon is {real_eps:.2f}', flush = True)

    torch.save(generator.state_dict(), EPS_PATH +'/models/generator.pth')
    torch.save(discriminator.state_dict(), EPS_PATH +'/models/discriminator.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain',action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--cuda',  default="cpu", type=str)
    parser.add_argument('--task', default='attention', type=str)    
    parser.add_argument('--ploss', default='3.0', type=float)
    parser.add_argument('--dloss', default='1.5', type=float)
    parser.add_argument('--data', default='geolife', type=str)
    parser.add_argument('--length', default=48, type=int)
    parser.add_argument('-e','--epsilon', default =1, type=float)
    parser.add_argument('-d', '--delta', default=1e-7, type=float)
    parser.add_argument('--skipm', action='store_true')
    parser.add_argument('-a', '--acum', default=1, type=int)
    opt = parser.parse_args()
    main(opt)