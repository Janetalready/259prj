import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from models import CNN
import torch.nn as nn
import torch
from tqdm import tqdm


class Bayes_Classifier:
    def __init__(self, seqname, labelname):
        with open(seqname,'rb') as fin:
            self.action_seqs = pickle.load(fin)
        with open(labelname,'rb') as fin:
            labels = pickle.load(fin)
        self.dictionary = {'watch_s':0, 'watch_s2':1, 'watch_c':2,
                           'play_s':3, 'play_s2':4, 'play_c':5,
                           'search_s':6, 'search_s2':7, 'search_c':8}
        self.temp_label = labels
        self.labels = []
        for label in labels:
            self.labels.append(self.dictionary[label])
        self.labels = np.array(self.labels)

    def get_bag_of_word(self, vocabulary=10):
        bag_of_words = []
        for idn, seq in enumerate(self.action_seqs):
            bag_of_word = np.zeros(vocabulary)
            unique_elements, counts_elements = np.unique(seq, return_counts=True)
            for idx, unique_element in enumerate(unique_elements):
                unique_element = int(unique_element)
                if unique_element == 0 or unique_element == 4 or unique_element == 8:
                    bag_of_word[0] += counts_elements[idx]
                elif unique_element<4:
                    bag_of_word[unique_element] += counts_elements[idx]
                elif unique_element>4 and unique_element<8:
                    bag_of_word[unique_element-1] += counts_elements[idx]
                else:
                    bag_of_word[unique_element-2] += counts_elements[idx]
            # print(bag_of_word)
            # print(self.temp_label[idn])
            # bag_of_word = bag_of_word[1:]
            # if np.linalg.norm(bag_of_word) == 0:
            #     print(bag_of_word)
            #     print(self.temp_label[idn])
            # bag_of_word = bag_of_word/np.linalg.norm(bag_of_word)
            bag_of_words.append(bag_of_word)
            # print(bag_of_word)
            # print(self.temp_label[idn])
        bag_of_words = np.array(bag_of_words)
        self.labels = self.labels.reshape((-1,1))
        self.bow_data = np.hstack([bag_of_words, self.labels])
        with open('bow_features.p','wb') as fout:
            pickle.dump(bag_of_words, fout)

    def add_noise(self, data, mag):
        noise = np.random.randn(data.shape)
        noise_data = data + mag*noise
        return noise_data

    def bayes(self, ratio = 0.8):
        train_length = int(self.bow_data.shape[0]* ratio)
        np.random.shuffle(self.bow_data)

        train_d = self.bow_data[:train_length,:-1]
        scores = dict()
        for mag in np.arange(1, 50, 2):
            train_d = self.add_noise(train_d, mag=mag)
            train_l = self.bow_data[:train_length,-1]
            test_d = self.bow_data[train_length:, :-1]
            test_l = self.bow_data[train_length:, -1]
            clf = MultinomialNB()
            clf.fit(train_d, train_l)
            predicted = clf.predict(test_d)
            assert test_l.shape == predicted.shape
            score = accuracy_score(test_l, predicted)
            print(score)
            scores[mag] = score
        print(scores)

    def logistic(self, ratio = 0.8):
        train_length = int(self.bow_data.shape[0] * ratio)
        np.random.shuffle(self.bow_data)
        train_d = self.bow_data[:train_length, :-1]
        train_l = self.labels[:train_length, -1]
        test_d = self.bow_data[train_length:, :-1]
        test_l = self.labels[train_length:, -1]
        clf = LogisticRegression(random_state=0, solver='saga', multi_class = 'multinomial').fit(train_d, train_l)
        predicted = clf.predict(test_d)
        assert test_l.shape == predicted.shape
        score = accuracy_score(test_l, predicted)
        print(score)

class GLOVE:
    def __init__(self, seqname, labelname, ratio=0.8):
        with open(seqname,'rb') as fin:
            self.action_seqs = pickle.load(fin)
        self.action_seqs = np.array(self.action_seqs)
        with open(labelname,'rb') as fin:
            labels = pickle.load(fin)
        self.dictionary = {'watch_s':0, 'watch_s2':1, 'watch_c':2,
                           'play_s':3, 'play_s2':4, 'play_c':5,
                           'search_s':6, 'search_s2':7, 'search_c':8}
        self.labels = []
        for label in labels:
            self.labels.append(self.dictionary[label])
        self.labels = np.array(self.labels)

        train_pos = np.random.choice(self.action_seqs.shape[0], int(self.action_seqs.shape[0]*ratio), replace=False)
        self.train_pos = train_pos
        total_pos = np.arange(0,self.action_seqs.shape[0])
        test_pos = set(total_pos) - set(train_pos)
        test_pos = np.array(list(test_pos))
        self.test_pos = test_pos
        self.train_seqs = self.action_seqs[train_pos]
        self.train_labels = self.labels[train_pos]
        self.test_seqs = self.action_seqs[test_pos]
        self.test_labels = self.labels[test_pos]

    # def build_matrix(self, vocabulary):
    #     adj_matrix = dict()
    #     for idx, seq in enumerate(self.action_seqs):
    #         for word in seq:
    #             ind = word[0]*4*4 + word[1]*4 + word[2]
    #             for adj_idx in range(-3, 3, 1):
    #                 if idx + adj_idx >=0 and idx + adj_idx <seq.shape[0]:
    #                     adj_word = seq[idx + adj_idx]
    #                     adj_ind = adj_word[0] * 4 * 4 + adj_word[1] * 4 + adj_word[2]
    #                     if not (int(ind), int(adj_ind)) in adj_matrix.keys():
    #                         adj_matrix[(int(ind), int(adj_ind))] = 0
    #                     adj_matrix[(int(ind), int(adj_ind))] += 1
    #     self.adj_matrix = adj_matrix
    def get_index(self, word):
        if word == 0 or word == 4 or word == 8:
            ind = 0
        elif word < 4:
            ind = word
        elif word > 4 and word < 8:
            ind = word - 1
        else:
            ind = word - 2
        return ind

    def build_matrix(self, vocabulary=10):
        adj_matrix = dict()
        adj_matrix_ = np.zeros((vocabulary, vocabulary))
        for idx, seq in enumerate(self.action_seqs):
            for word in seq:
                ind = self.get_index(word)
                for adj_idx in range(-3, 3, 1):
                    if idx + adj_idx >=0 and idx + adj_idx <seq.shape[0]:
                        adj_word = seq[idx + adj_idx]
                        adj_ind = self.get_index(adj_word)
                        if not (int(ind), int(adj_ind)) in adj_matrix.keys():
                            adj_matrix[(int(ind), int(adj_ind))] = 0
                        adj_matrix[(int(ind), int(adj_ind))] += 1
                        adj_matrix_[int(ind), int(adj_ind)] += 1
        self.adj_matrix = adj_matrix

    def run_iter(self, vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):

        global_cost = 0
        np.random.shuffle(data)

        for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
             gradsq_b_main, gradsq_b_context, cooccurrence) in data:
            weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

            cost_inner = (v_main.dot(v_context)
                          + b_main[0] + b_context[0]
                          - np.log(cooccurrence))

            cost = weight * (cost_inner ** 2)

            global_cost += 0.5 * cost

            grad_main = weight * cost_inner * v_context
            grad_context = weight * cost_inner * v_main

            grad_bias_main = weight * cost_inner
            grad_bias_context = weight * cost_inner

            v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
            v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

            b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
            b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

            gradsq_W_main += np.square(grad_main)
            gradsq_W_context += np.square(grad_context)
            gradsq_b_main += grad_bias_main ** 2
            gradsq_b_context += grad_bias_context ** 2

        return global_cost

    def train_glove(self, vocab, cooccurrences, iter_callback=None, vector_size=100,
                    iterations=50, **kwargs):

        vocab_size = vocab
        W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)
        biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

        gradient_squared = np.ones((vocab_size * 2, vector_size),
                                   dtype=np.float64)
        gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

        data = [(W[i_main], W[i_context + vocab_size],
                 biases[i_main: i_main + 1],
                 biases[i_context + vocab_size: i_context + vocab_size + 1],
                 gradient_squared[i_main], gradient_squared[i_context + vocab_size],
                 gradient_squared_biases[i_main: i_main + 1],
                 gradient_squared_biases[i_context + vocab_size
                                         : i_context + vocab_size + 1],
                 cooccurrence)
                for (i_main, i_context), cooccurrence in cooccurrences.items()]

        for i in range(iterations):
            print(["Beginning iteration %i..", i])

            cost = self.run_iter(vocab, data, **kwargs)

            print(["Done (cost %f)", cost])

        self.embedding = W
        with open('glove_embedding.p','wb') as fin:
            pickle.dump(W, fin)

    def load_embedding(self, filename):
        with open(filename,'rb') as fin:
            self.embedding = pickle.load(fin)

    def add_noise(self, data, mag):
        noise = np.random.randn(data.shape[0], data.shape[1])
        noise_data = data + mag * noise
        return noise_data

    def get_embedding(self, seqs):
        embed_seqs = []
        max_len = 0
        batch_size = len(seqs)
        for idn, seq in enumerate(seqs):
            embed_seq = np.empty((0,self.embedding.shape[1]))
            if seq.shape[0] > max_len:
                max_len = seq.shape[0]
            for word in seq:
                idx = self.get_index(word)
                embed_seq = np.vstack([embed_seq, self.embedding[int(idx)]])
            embed_seqs.append(embed_seq)

        embed_seqs_batch = np.zeros((batch_size, max_len, self.embedding.shape[1]))
        for idx, seq in enumerate(embed_seqs):
            seq = self.add_noise(seq, mag=1)
            embed_seqs_batch[idx,:seq.shape[0],:] = seq
        embed_seqs_batch = embed_seqs_batch.reshape((batch_size, 1, max_len, self.embedding.shape[1]))
        return embed_seqs_batch

    def test(self, net, batch_size, test_data, test_label):
        correct = 0
        total = 0
        net.eval()
        pbar = tqdm(range(0, test_data.shape[0], batch_size))
        for batch_num in pbar:
            total += 1
            if batch_num + batch_size > test_data.shape[0]:
                end = test_data.shape[0]
            else:
                end = batch_num + batch_size
            raw_inputs, actual_val = test_data[batch_num:end], test_label[batch_num:end]
            inputs_ = self.get_embedding(raw_inputs)
            # perform classification
            inputs = torch.from_numpy(inputs_).float().cuda()
            actual_val = torch.from_numpy(actual_val).cuda()
            predicted_val = net(torch.autograd.Variable(inputs))
            # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
            predicted_val = predicted_val.data
            max_score, idx = torch.max(predicted_val, 1)
            assert idx.shape==actual_val.shape
            # compare it with actual value and estimate accuracy
            correct += (idx == actual_val).sum()
            pbar.set_description("processing batch %s" % str(batch_num))
        print("Classifier Accuracy: ", float(correct.cpu().numpy()) / test_data.shape[0])
        return float(correct.cpu().numpy()) / test_data.shape[0]



    def bilstm_train(self, numEpochs, batch_size, save_file, lr):
        print('training .....')

        # set up loss function -- 'SVM Loss' a.k.a ''Cross-Entropy Loss
        loss_func = nn.CrossEntropyLoss()
        net = CNN(embed_dim=100)
        # net.load_state_dict(torch.load('model_50.pth'))
        # SGD used for optimization, momentum update used as parameter update
        optimization = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        net.cuda()
        loss_func.cuda()
        train_losses = []
        test_losses = []
        for epoch in range(0,numEpochs):

            # training set -- perform model training
            epoch_training_loss = 0.0
            num_batches = 0
            pbar = tqdm(range(0, len(self.train_seqs), batch_size))
            for batch_num in pbar:  # 'enumerate' is a super helpful function
                # split training data into inputs and labels
                if batch_num+batch_size>len(self.train_seqs):
                    end = len(self.action_seqs)
                else:
                    end = batch_num+batch_size
                raw_inputs, labels_ = self.train_seqs[batch_num:end], self.train_labels[batch_num:end]  # 'training_batch' is a list
                inputs_ = self.get_embedding(raw_inputs)
                inputs = torch.from_numpy(inputs_).float().cuda()
                labels = torch.from_numpy(labels_).cuda()
                # wrap data in 'Variable'
                inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                # Make gradients zero for parameters 'W', 'b'
                optimization.zero_grad()
                # forward, backward pass with parameter update
                forward_output = net(inputs)
                loss = loss_func(forward_output, labels)
                loss.backward()
                optimization.step()
                # calculating loss
                epoch_training_loss += loss.data.item()
                num_batches += 1
                # print(loss.data.item())
                pbar.set_description("processing batch %s" % str(batch_num))
            print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)
            # train_loss = self.test(net, batch_size=256, test_data=self.train_seqs, test_label=self.train_labels)
            test_loss = self.test(net, batch_size=256, test_data=self.test_seqs, test_label=self.test_labels)
            # train_losses.append(train_loss)
            test_losses.append(test_loss)
            # if epoch%10 == 0:
            #     save_path = save_file+'model3_' +str(epoch)+'.pth'
            #     torch.save(net.state_dict(), save_path)
        # with open('train_loss_1.p','wb') as fin:
        #     pickle.dump(train_losses,fin)
        #     fin.close()
        with open('test_loss_1.p','wb') as fin:
            pickle.dump(test_losses,fin)
            fin.close()






if __name__ == '__main__':
    # bow_classifier = Bayes_Classifier('action_seqs.p','labels.p')
    # bow_classifier.get_bag_of_word()
    # bow_classifier.bayes()

    glove = GLOVE('action_seqs.p','labels.p')
    # glove.build_matrix(10)
    # glove.train_glove(10, glove.adj_matrix, iterations=150)
    glove.load_embedding('glove_embedding.p')
    glove.bilstm_train(numEpochs=50, batch_size=256, save_file='./', lr=0.01)

