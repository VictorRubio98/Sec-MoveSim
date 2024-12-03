from generator import *
from discriminator import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class SecMoveSim(nn.Module):
    def __init__(
            self,
            total_locations=8606,
            embedding_net=None,
            loc_embedding_dim=256,
            tim_embedding_dim=16,
            hidden_dim=64,
            bidirectional=False,
            data='geolife',
            device=None,
            function=False,
            make_private=False):
        """

        :param total_locations:
        :param embedding_net:
        :param embedding_dim:
        :param hidden_dim:
        :param bidirectional:
        :param cuda:
        :param starting_sample:
        :param starting_dist:
        """
        self.device = device
        self.total_locations = total_locations
        self.data = data
        self.function = function
        self.make_private = make_private
        self.generator = ATGenerator(device=device,total_locations=total_locations,starting_sample='real', starting_dist=np.load(f'../data/{data}/start.npy'),data=data, 
                                     make_private=make_private, function=function, hidden_dim=hidden_dim, embedding_net=embedding_net, loc_embedding_dim=loc_embedding_dim, tim_embedding_dim=tim_embedding_dim, bidirectional=bidirectional)
        self.discriminator = Discriminator(total_locations=total_locations)