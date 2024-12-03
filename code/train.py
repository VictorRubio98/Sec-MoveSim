# encoding: utf-8
import torch
import numpy as np
from opacus.data_loader import DPDataLoader
from torch.optim import Optimizer

def generate_samples(model, batch_size, seq_len, generated_num, output_file:str, inference:bool = False):
    samples = []
    noises = []
    for _ in range(int(generated_num / batch_size)):
        if inference:
            sample, noise = model.sample(
                batch_size, seq_len, inference=inference)
            sample = sample.cpu().data.numpy().tolist()
            noise = noise.cpu().data.numpy().tolist()
            noises.extend(noise)
        else:
            sample = model.sample(
                batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    if inference:
        with open(output_file.replace('samples','noise'), 'w') as fout:
            for noise in noises:
                noise = ' '.join([str(n) for n in noise])
                fout.write('%s\n' % noise)



def generate_samples_to_mem(model, batch_size, seq_len, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)


def pretrain_model(
        name,
        pre_epochs,
        model,
        data_iter,
        criterion,
        optimizer,
        batch_size,
        device=None,
        grad_acum = 1):
    lloss = 0.
    for epoch in range(pre_epochs):
        loss = train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device, grad_acum)
        print('Epoch [%d], loss: %f' % (epoch + 1, loss), flush=True)
        if loss < 0.01 or 0 < lloss - loss < 0.01:
            print("early stop at epoch %d" % (epoch + 1), flush=True)
            break


def train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device=None, grad_acum=1):
    total_loss = 0.
    criterion = criterion.to(device)
        
    for i, (data, target) in enumerate(data_iter):
        data = torch.LongTensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        target = target.contiguous().view(-1)
        if name == "G":
            tim = torch.LongTensor([i%24 for i in range(data.shape[1])]).to(device)
            tim = tim.repeat(data.shape[0]).reshape(data.shape[0], -1)
            pred = model(data, tim)
        else:
            pred = model(data)
        loss = criterion(pred, target)
        loss = loss / grad_acum
        total_loss += loss.item()
        loss.backward()
        if ((i + 1) % grad_acum == 0) or (i + 1 == len(data_iter)):
            optimizer.step()

            optimizer.zero_grad()
    if data_iter.__class__ is not DPDataLoader: # Use reset when using normal dataloader
        data_iter.reset()
    return total_loss / (i + 1)