import numpy as np
import pickle

class Data_Generator:
    def __init__(self, object, type, pos, prefix = './fake_data/'):
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
        end_pos = np.random.randint(start_pos, length)
        watch_seq[start_pos:end_pos] = 1
        return watch_seq

    def generate_play_seq(self,min_seq=100, max_seq=300):
        length = np.random.randint(min_seq, max_seq)
        play_seq = np.zeros(length*2)
        start_pos = np.random.randint(0, length / 2)
        end_pos = np.random.randint(start_pos, length)
        gaze_pos = np.arange(start_pos, end_pos*2+1, 2)
        play_pos = np.arange(start_pos+1, end_pos * 2+1, 2)
        play_seq[gaze_pos] = 1
        play_seq[play_pos] = 2
        return play_seq

    def generate_watch(self):
        length = np.random.randint(300, 800)
        obj_sequence = np.zeros(length)
        start_pos = np.random.randint(0, length/2)
        end_pos = np.random.randint(start_pos, length)
        obj_sequence[start_pos:end_pos] = 1
        with open(self.obj_path, 'wb') as fin:
            pickle.dump(obj_sequence, fin)

        other1_sequence = np.zeros(length)
        invisible = np.random.randint(0,2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other1_sequence[positions] = 3
        with open(self.other1_path, 'wb') as fin:
            pickle.dump(other1_sequence, fin)

        other2_sequence = np.zeros(length)
        invisible = np.random.randint(0, 2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other2_sequence[positions] = 3
        with open(self.other2_path, 'wb') as fin:
            pickle.dump(other2_sequence, fin)

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
        with open(self.obj_path, 'wb') as fin:
            pickle.dump(play_watch_seq, fin)

        length = play_watch_seq.shape[0]
        other1_sequence = np.zeros(length)
        invisible = np.random.randint(0, 2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other1_sequence[positions] = 3
        with open(self.other1_path, 'wb') as fin:
            pickle.dump(other1_sequence, fin)

        other2_sequence = np.zeros(length)
        invisible = np.random.randint(0, 2)
        if invisible:
            positions = self.generate_invisible_other(length)
            other2_sequence[positions] = 3
        with open(self.other2_path, 'wb') as fin:
            pickle.dump(other2_sequence, fin)

    def generate_search(self):
        play = np.random.randint(0,2)
        print(play)
        if play:
            play_seq = self.generate_play_seq(100,200)
        length = np.random.randint(300, 800)
        obj_sequence = np.zeros(length)
        start_pos = np.random.randint(0, length / 2)
        end_pos = np.random.randint(start_pos, length)
        obj_sequence[start_pos:end_pos] = 3
        if play:
            obj_sequence = np.concatenate([play_seq, obj_sequence])
        with open(self.obj_path, 'wb') as fin:
            pickle.dump(obj_sequence, fin)
        print(obj_sequence)

        other1_sequence = np.zeros(obj_sequence.shape[0])
        if play:
            invisible = np.random.randint(0, 2)
            if invisible:
                positions = self.generate_invisible_other(length)
                other1_sequence[positions] = 3
        print(other1_sequence)
        with open(self.other1_path, 'wb') as fin:
            pickle.dump(other1_sequence, fin)

        other2_sequence = np.zeros(obj_sequence.shape[0])
        if play:
            invisible = np.random.randint(0, 2)
            if invisible:
                positions = self.generate_invisible_other(length)
                other2_sequence[positions] = 3
        print(other2_sequence)
        with open(self.other2_path, 'wb') as fin:
            pickle.dump(other2_sequence, fin)

if __name__ == '__main__':
    data_generator = Data_Generator(object='c', type='search', pos='p8')
    data_generator.generate_search()
