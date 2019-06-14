import numpy as np
import pickle

class Data_Generator:
    def __init__(self, object, type, pos, prefix = '/home/shuwen/ucla/259/proj/raw_data3/'):
        total_object = {'s':'sphere','c':'cylinder','s2':'sphere2'}
        self.obj_path = prefix+type+'/'+total_object[object]+'_'+type+'_'+object+'_'+pos+'.p'
        flag = 0
        for object_ in total_object.keys():
            if (not object_ == object):
                if flag == 0:
                    self.other1_path = prefix+type+'/'+total_object[object_]+'_'\
                                       +type+'_'+object+'_'+pos+'.p'
                    flag += 1
                else:
                    self.other2_path = prefix + type + '/' + total_object[object_]+'_' \
                                       + type + '_' + object + '_' + pos + '.p'

    def generate_invisible_other(self,length):
        invisible_length = np.random.randint(0, length)
        positions = np.random.randint(0, length, invisible_length)
        return positions

    def generate_watch_seq(self, min_seq=100, max_seq=200):
        length = np.random.randint(min_seq, max_seq)
        watch_seq = np.zeros(length)
        start_pos = np.random.randint(0, length / 2)
        end_pos = np.random.randint(start_pos+length/4, length)
        watch_seq[start_pos:end_pos] = 1
        return watch_seq

    def generate_play_seq(self,min_seq=100, max_seq=300):
        length = np.random.randint(min_seq, max_seq)
        play_seq = np.zeros(length*2)
        start_pos = np.random.randint(0, length / 2)
        end_pos = np.random.randint(start_pos+length/4, length)
        gaze_pos = np.arange(start_pos, end_pos*2+1, 2)
        play_pos = np.arange(start_pos+1, end_pos * 2+1, 2)
        play_seq[gaze_pos] = 1
        play_seq[play_pos] = 2
        return play_seq

    def generate_watch(self):
        length = np.random.randint(300, 800)
        obj_sequence = np.zeros(length)
        start_pos = np.random.randint(0, length/2)
        end_pos = np.random.randint(start_pos+length/4, length)
        obj_sequence[start_pos:end_pos] = 1
        if np.linalg.norm(obj_sequence) == 0:
            print([start_pos, end_pos])
        # with open(self.obj_path, 'wb') as fin:
        #     pickle.dump(obj_sequence, fin)
        #     fin.close()

        other1_sequence = np.zeros(length)
        invisible = np.random.randint(0,2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other1_sequence[positions] = 3
        # with open(self.other1_path, 'wb') as fin:
        #     pickle.dump(other1_sequence, fin)
        #     fin.close()

        other2_sequence = np.zeros(length)
        invisible = np.random.randint(0, 2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other2_sequence[positions] = 3
        # with open(self.other2_path, 'wb') as fin:
        #     pickle.dump(other2_sequence, fin)
        #     fin.close()
        return obj_sequence, other1_sequence, other2_sequence

    def generate_play(self):
        watch_seq = self.generate_watch_seq()
        play_seq = self.generate_play_seq()
        interval = np.random.randint(0,3)
        play_watch_seq = []
        if interval == 0:
            play_watch_seq.append(watch_seq)
            play_watch_seq.append(play_seq)
        elif interval == 1:
            length = play_seq.shape[0]
            pivot = np.random.randint(0, length)
            play_watch_seq.append(play_seq[:pivot])
            play_watch_seq.append(watch_seq)
            play_watch_seq.append(play_seq[pivot:])
        else:
            play_watch_seq.append(play_seq)
            play_watch_seq.append(watch_seq)
        play_watch_seq = np.array(play_watch_seq)
        play_watch_seq_ = np.empty((1, 0))
        for seq in play_watch_seq:
            seq = seq.reshape((1, -1))
            play_watch_seq_ = np.hstack([play_watch_seq_, seq])
        play_watch_seq_ = play_watch_seq_.reshape(-1)
        # with open(self.obj_path, 'wb') as fin:
        #     pickle.dump(play_watch_seq_, fin)
        #     fin.close()

        length = play_watch_seq_.shape[0]
        other1_sequence = np.zeros(length)
        invisible = np.random.randint(0, 2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other1_sequence[positions] = 3
        # with open(self.other1_path, 'wb') as fin:
        #     pickle.dump(other1_sequence, fin)
        #     fin.close()

        other2_sequence = np.zeros(length)
        invisible = np.random.randint(0, 2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other2_sequence[positions] = 3
        # with open(self.other2_path, 'wb') as fin:
        #     pickle.dump(other2_sequence, fin)
        #     fin.close()
        return play_watch_seq_, other1_sequence, other2_sequence

    def generate_search(self):
        play = np.random.randint(0,2)
        if play:
            play_seq = self.generate_play_seq(100,200)
        length = np.random.randint(300, 800)
        obj_sequence = np.zeros(length)
        start_pos = np.random.randint(0, length / 2)
        end_pos = np.random.randint(start_pos+length/4, length)
        obj_sequence[start_pos:end_pos] = 3
        if play:
            obj_sequence = np.concatenate([play_seq, obj_sequence])
        # with open(self.obj_path, 'wb') as fin:
        #     pickle.dump(obj_sequence, fin)
        #     fin.close()
        # print(obj_sequence)

        other1_sequence = np.zeros(obj_sequence.shape[0])
        if play:
            invisible = np.random.randint(0, 2)
            if invisible:
                positions = self.generate_invisible_other(length)
                other1_sequence[positions] = 3
        # print(other1_sequence)
        # with open(self.other1_path, 'wb') as fin:
        #     pickle.dump(other1_sequence, fin)
        #     fin.close()

        other2_sequence = np.zeros(obj_sequence.shape[0])
        if play:
            invisible = np.random.randint(0, 2)
            if invisible:
                positions = self.generate_invisible_other(length)
                other2_sequence[positions] = 3
        # print(other2_sequence)
        # with open(self.other2_path, 'wb') as fin:
        #     pickle.dump(other2_sequence, fin)
        #     fin.close()
        return obj_sequence, other1_sequence, other2_sequence

if __name__ == '__main__':
    obj_seqs = []
    other1_seqs = []
    other2_seqs = []
    for i in range(1, 1001):
        data_generator = Data_Generator(object='c', type='search', pos='p'+str(i))
        obj_path = data_generator.obj_path
        other1_path = data_generator.other1_path
        other2_path  =data_generator.other2_path
        obj_sequence, other1_sequence, other2_sequence = data_generator.generate_search()
        obj_seqs.append(obj_sequence)
        other1_seqs.append(other1_sequence)
        other2_seqs.append(other2_sequence)
    obj_seqs = np.array(obj_seqs)
    other1_seqs = np.array(other1_seqs)
    other2_seqs = np.array(other2_seqs)
    with open(obj_path,'wb') as fin:
        pickle.dump(obj_seqs, fin)
        fin.close()
    with open(other1_path,'wb') as fin:
        pickle.dump(other1_seqs, fin)
        fin.close()
    with open(other2_path,'wb') as fin:
        pickle.dump(other2_seqs, fin)
        fin.close()

