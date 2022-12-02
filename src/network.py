import torch
import torch.nn as nn
import math

class DocumentNetFc(nn.Module):
    def __init__(self, bloom_filter_size, numLayers, hash_bit, layer_size):
        super(DocumentNetFc, self).__init__()
        self.feature_layers = nn.Sequential()

        for i in range(numLayers):
            if i == 0:
                self.feature_layers.add_module("fc_layer_" + str(i), nn.Linear(bloom_filter_size, layer_size))
            else:
                self.feature_layers.add_module("fc_layer_" + str(i), nn.Linear(layer_size, layer_size))
            self.feature_layers.add_module("relu_layer_" + str(i), nn.ReLU())

        hash_layer_size = 0

        if numLayers == 0:
            hash_layer_size = bloom_filter_size
        else:
            hash_layer_size = layer_size
        
        self.hash_layer = nn.Linear(hash_layer_size, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale
    
    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y