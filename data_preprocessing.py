import pickle
import numpy as np
from itertools import product

class Data_Preprossor:
    def __init__(self, object, type, pos, prefix = './raw_data/'):
        sphere_file = open(prefix+type+'/'+'sphere_'+type+'_'
                           +object+'_'+pos+'.p','rb')
        sphere2_file = open(prefix + type + '/' + 'sphere2_'
                            + type + '_' + object + '_' + pos + '.p', 'rb')
        cylinder_file = open(prefix + type + '/' + 'cylinder_'
                           + type + '_' + object + '_' + pos + '.p', 'rb')
        self.type = type
        self.object = object
        self.sphere_seq = pickle.load(sphere_file)
        self.sphere2_seq = pickle.load(sphere2_file)
        self.cylinder_seq = pickle.load(cylinder_file)
        min_length = min(self.sphere_seq.shape[0], self.sphere2_seq.shape[0], self.cylinder_seq.shape[0])
        self.sequence = np.vstack((self.sphere_seq[:min_length],
                                   self.sphere2_seq[:min_length]+4,
                                   self.cylinder_seq[:min_length]+8))

    def get_action_sequence(self):
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
        return action_seq, label

if __name__ == '__main__':
    raw_data_list = list(product(['s','s2','c'],
                                 ['watch','play','search'],
                                 ['p1']))
    action_seqs = []
    labels = []
    for (object, type, pos) in raw_data_list:
        data_preprossor = Data_Preprossor(object, type, pos)
        action_seq, label = data_preprossor.get_action_sequence()
        print(action_seq)
        print(label)
        action_seqs.append(action_seq)
        labels.append(label)
    with open('action_seqs.p','wb') as fin:
        pickle.dump(action_seqs, fin)
    with open('labels.p','wb') as fin:
        pickle.dump(labels, fin)
