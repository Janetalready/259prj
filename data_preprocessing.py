import pickle
import numpy as np
from itertools import product

class Data_Preprossor:
    def __init__(self, object, type, pos, prefix = './raw_data3/'):
        sphere_file = open(prefix+type+'/'+'sphere_'+type+'_'
                           +object+'_'+pos+'.p','rb')
        sphere2_file = open(prefix + type + '/' + 'sphere2_'
                            + type + '_' + object + '_' + pos + '.p', 'rb')
        cylinder_file = open(prefix + type + '/' + 'cylinder_'
                           + type + '_' + object + '_' + pos + '.p', 'rb')
        self.type = type
        self.object = object
        self.sphere_seqs = pickle.load(sphere_file)
        self.sphere2_seqs = pickle.load(sphere2_file)
        self.cylinder_seqs = pickle.load(cylinder_file)
        # min_length = min(self.sphere_seq.shape[0], self.sphere2_seq.shape[0], self.cylinder_seq.shape[0])
        # print(sphere_file)
        # print(self.sphere_seq.shape)
        # self.sequence = np.vstack((self.sphere_seq[:min_length],
        #                            self.sphere2_seq[:min_length]+4,
        #                            self.cylinder_seq[:min_length]+8))

    def get_action_seq(self):
        action_seqs = []
        labels = []
        for idx, sphere_seq in enumerate(self.sphere_seqs):
            sphere2_seq = self.sphere2_seqs[idx]
            cylinder_seq = self.cylinder_seqs[idx]
            min_length = min(sphere_seq.shape[0], sphere2_seq.shape[0], cylinder_seq.shape[0])
            self.sequence = np.vstack((sphere_seq[:min_length],
                                       sphere2_seq[:min_length]+4,
                                       cylinder_seq[:min_length]+8))
            action_seq = []
            for i in range(self.sequence.shape[1]):
                flag = 0
                for j in range(self.sequence.shape[0]):
                    if not (self.sequence[j,i] == 0 or self.sequence[j,i] == 4
                            or self.sequence[j,i] == 8):
                        action_seq.append(self.sequence[j,i])
                        flag = 1
                if flag == 0:
                    action_seq.append(0)
            action_seq = np.array(action_seq)
            label = self.type + '_' + self.object
            action_seqs.append(action_seq)
            labels.append(label)
        return action_seqs, labels

    def get_action_seq_3(self):
        action_seqs = []
        labels = []
        for idx, sphere_seq in enumerate(self.sphere_seqs):
            sphere2_seq = self.sphere2_seqs[idx]
            cylinder_seq = self.cylinder_seqs[idx]
            min_length = min(sphere_seq.shape[0], sphere2_seq.shape[0], cylinder_seq.shape[0])
            self.sequence = np.vstack((sphere_seq[:min_length],
                                       sphere2_seq[:min_length],
                                       cylinder_seq[:min_length]))
            label = self.type + '_' + self.object
            action_seqs.append(self.sequence.T)
            labels.append(label)
        return action_seqs, labels

if __name__ == '__main__':
    raw_data_list = list(product(['s','s2','c'],
                                 ['watch','play','search'],
                                 ['p1000']))
    action_seqs = []
    labels = []
    idx = 0
    for (object, type, pos) in raw_data_list:
        data_preprossor = Data_Preprossor(object, type, pos)
        action_seq, label = data_preprossor.get_action_seq()
        action_seqs += action_seq
        labels += label
    with open('action_seqs.p','wb') as fin:
        pickle.dump(action_seqs, fin)
    with open('labels.p','wb') as fin:
        pickle.dump(labels, fin)
