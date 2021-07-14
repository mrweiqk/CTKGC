import numpy as np
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn

import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class TWST(torch.nn.Module):


    def __init__(self,  d, d1, d2, **kwargs):
        super(TWST, self).__init__()


        print("d=,d1=,d2=,",d,d1,d2)
        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)

        num_entities=len(d.entities)
        num_relations=len(d.relations)

        #hidden_size=5120

        input_drop=0.2
        hidden_drop=0.5
        #hidden_drop = 0.1
        feat_drop=0.2
        embedding_shape1=20
        embedding_dim=200
        self.embedding_dim = 200
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = 20
        self.emb_dim2 = embedding_dim // self.emb_dim1

        atthidden_drop = 0.1
        self.atthidden_drop=torch.nn.Dropout(hidden_drop)
        ##self.atthidden_drop=torch.nn.Dropout(hidden_drop)
        con_num = 32
        con_1 = 3
        con_2 = 200
        hidden_size = con_num *(embedding_dim-con_1 +1) * (embedding_dim-con_2 +1)
        self.conv1 = torch.nn.Conv2d(1, con_num, (con_1, con_2), 1, 0, bias=True)# 输入1维 输出32维  卷积核3*3 步长1 池化0
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(con_num)
        #self.bn2 = torch.nn.BatchNorm1d(embedding_dim*2)
        #self.bn2 = torch.nn.BatchNorm1d(9728)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)


        self.attbn2=torch.nn.BatchNorm1d(embedding_dim)

        #self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        #self.fc = torch.nn.Linear(hidden_size, embedding_dim)
        self.fc = torch.nn.Linear(hidden_size, embedding_dim)
        #print(num_entities, num_relations)

        hidden_size1=9728
        #self.fc2=torch.nn.Linear(hidden_size1,embedding_dim)
        #self.fc2 = torch.nn.Linear(400, embedding_dim)
        self.fc2 = torch.nn.Linear(800, embedding_dim)



        stride=1
        downsample = None
        #planes=200
        #planes=4
        planes=32
        #self.ca = ChannelAttention(planes)
        #self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)



    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)





    def forward(self, e1_idx, r_idx):
        #print("forward+++++++++++++")

        e1 = self.E(e1_idx)# 128*200
        e1_embedded=e1.view(-1,self.embedding_dim,1)

        r = self.R(r_idx)
        rel_embedded=r.view(-1, 1, self.embedding_dim)

        stacked_inputs = torch.bmm(e1_embedded,rel_embedded)#128*50*50
        stacked_inputs = stacked_inputs.view(-1,1,self.embedding_dim,self.embedding_dim)#128*1*50*50
        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)#128*1*50*50

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)#128*1*200*200
        x = self.conv1(x)# 128*32*38*8##128*32*198*198
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)                     #128*32*198*1
        # #*******
        # x = self.ca(x) * x
        # #*******
        x = x.view(x.shape[0], -1)# 128* 9728
        #print("x=x.view",x.shape)

        x = self.fc(x)# 128*800#输入128*9728  隐藏层9728 输出层800## f

        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print("x.shape",x.shape)


        x=torch.mm(x,self.E.weight.transpose(1,0))#128*14541
        #x = torch.mm(x, self.emb_e.weight.transpose(1, 0))

        #print("x_pred.shape")
        #print(x.shape)


        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        #print("pred.shape")
        #print(pred.shape)

        return pred
