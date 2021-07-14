from load_data import *
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
import configparser
import json
import networkx as nx
from max import *

class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.,neb_size=20, max_size=70, json_file='ee.json'):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.json_file = json_file
        self.neb_size = neb_size
        self.max_size = max_size
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "batch_size":batch_size,
                       "neb_size": neb_size}

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs



    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab



    def get_batch(self, er_vocab, er_vocab_pairs, idx):

        ##实体—关系对 batch
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        if (len(batch) != self.batch_size):
            batch = er_vocab_pairs[-self.batch_size:]
        ##标签,正确候选集标签1,错误候选集标签0,是一个0-1的分布向量
        targets = np.zeros((len(batch), len(d.entities)))

        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.

        targets = torch.FloatTensor(targets)

        if self.cuda:
            targets = targets.cuda()

        return np.array(batch), targets


    def evaluate(self, model, data,max):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):

            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])#.long()
            r_idx = torch.tensor(data_batch[:, 1])#.long()
            e2_idx = torch.tensor(data_batch[:, 2])#.long()

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        with open("test_results.txt", "a") as g:
            g.write("Test"+"\n")
            g.write("Number of data points: %d" % len(test_data_idxs)+"\n")
            g.write('Hits @10: {0}'.format(np.mean(hits[9]))+"\n")
            g.write('Hits @3: {0}'.format(np.mean(hits[2]))+"\n")
            g.write('Hits @1: {0}'.format(np.mean(hits[1]))+"\n")
            g.write('Mean rank: {0}'.format(np.mean(ranks))+"\n")
            g.write('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks)))+"\n")
        g.close()

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        if (max.MRR < np.mean(1. / np.array(ranks))):
            print('-----------------------------------------')
            max.MRR = np.mean(1. / np.array(ranks))
            max.MR=np.mean(ranks)
            max.hit1=np.mean(hits[0])
            max.hit3=np.mean(hits[2])
            max.hit10=np.mean(hits[9])
            print('Max Hits @10: {0}'.format(np.mean(hits[9])))
            print('Max Hits @3: {0}'.format(np.mean(hits[2])))
            print('Max Hits @1: {0}'.format(np.mean(hits[0])))
            print('Max Mean rank: {0}'.format(np.mean(ranks)))
            print('Max Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self):
        print("Training the TWST model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TWST(d=d, d1=self.ent_vec_dim, d2=self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        max = Max()
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs)//2, self.batch_size):

                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])#.long()
                r_idx = torch.tensor(data_batch[:, 1])#.long()

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda().long()

                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            with open("test_results.txt", "a") as g:
                g.write("Train" + "\n")
                g.write("it=" + str(it) + "\n")
            g.close()
            print('train_time',time.time() - start_train)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data,max)
                if not it % 2:
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data,max)
                    print('test_time:',time.time() - start_test)


if __name__ == '__main__':
    """
    --dataset: Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.
    --num_iterations: Number of iterations.
    --batch_size: Batch size.
    --lr: Learning rate.
    --dr: Decay rate.
    --edim: Entity embedding dimensionality.
    --rdim: Relation embedding dimensionality.
    --cuda: Whether to use cuda (GPU) or not (CPU).
    --input_dropout: Input layer dropout.
    --hidden_dropout1: Dropout after the first hidden layer.
    --hidden_dropout2: Dropout after the second hidden layer.
    --label_smoothing: Amount of label smoothing.
    
    --features: Graph Entity node features.
    --gcn_hidden: GCN hidden layer num.
    --neb_size: h neb node size.
    --weight_decay: Weight decay (L2 loss on parameters).
    --h_r neb max node size: size the max_size node.
    """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = configparser.ConfigParser()
    config.read('config.ini')


    # dataset = 'DBpedia50'
    # dataset = 'YAGO3-10'
    dataset = 'FB15k'
    #dataset = 'FB15k-237'
    #dataset = 'WN18'
    #dataset = 'WN18RR'
    #dataset = 'KINSHIP'
    #dataset = 'UMLS'
    num_iterations = int(config.get(section='globle', option='num_iterations'))
    batch_size = int(config.get(section='globle', option='batch_size'))
    lr = float(config.get(section=dataset, option='lr'))
    dr = float(config.get(section=dataset, option='dr'))
    edim = int(config.get(section=dataset, option='edim'))
    rdim = int(config.get(section=dataset, option='rdim'))
    cuda = True
    input_dropout = float(config.get(section=dataset, option='input_dropout'))
    hidden_dropout1 = float(config.get(section=dataset, option='hidden_dropout1'))
    hidden_dropout2 = float(config.get(section=dataset, option='hidden_dropout2'))
    label_smoothing = float(config.get(section=dataset, option='label_smoothing'))

    features = int(config.get(section='globle', option='features'))
    gcn_hidden = int(config.get(section='globle', option='gcn_hidden'))
    gcn_dropout = float(config.get(section='globle', option='gcn_dropout'))
    neb_size = int(config.get(section='globle', option='neb_size'))
    weight_decay = float(config.get(section='globle', option='weight_decay'))
    max_size = int(config.get(section='globle', option='max_size'))
    json_file = str(config.get(section='globle', option='json_file'))
    json_file = dataset+json_file

    para_dict = {'dataset':dataset,'num_iterations':num_iterations,
                      'batch_size':batch_size,'lr':lr,'dr':dr,
                      'edim':edim,'rdim':rdim,'input_dropout':input_dropout,
                      'hidden_dropout1':hidden_dropout1,'hidden_dropout2':hidden_dropout2,
                      'label_smoothing':label_smoothing,'features':features,
                 'gcn_hidden':gcn_hidden,'neb_size':neb_size,'weight_decay':weight_decay,
                 'max_size':max_size,'json_file':json_file,'gcn_dropout':gcn_dropout,
                 }

    for k in para_dict:
        print(k, para_dict[k])


    dir = os.path.abspath(os.path.dirname(os.getcwd()))
    print(dir)
    data_dir = dir+"/data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 17
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=num_iterations, batch_size=batch_size, learning_rate=lr,
                            decay_rate=dr, ent_vec_dim=edim, rel_vec_dim=rdim, cuda=cuda,
                            input_dropout=input_dropout, hidden_dropout1=hidden_dropout1,
                            hidden_dropout2=hidden_dropout2, label_smoothing=label_smoothing,
                            neb_size=neb_size, max_size=max_size, json_file=json_file)
    experiment.train_and_eval()
