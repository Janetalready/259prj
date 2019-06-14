import pickle
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, bowfile, labelname):
        with open(bowfile, 'rb') as fin:
            self.bows = pickle.load(fin)
        with open(labelname,'rb') as fin:
            self.labels = pickle.load(fin)
        self.dictionary = {'watch_s': 'watch_A', 'watch_s2': 'watch_B', 'watch_c': 'watch_C',
                           'play_s': 'play_A', 'play_s2': 'play_B', 'play_c': 'play_C',
                           'search_s': 'search_A', 'search_s2': 'search_B', 'search_c': 'search_C'}

    def word_distribution(self):
        distri = dict()
        for i in range(self.bows.shape[0]):
            label = self.labels[i]
            if not label in distri.keys():
                distri[label] = self.bows[i]
            else:
                distri[label] += self.bows[i]
        labels = distri.keys()
        for i in range(len(labels)):
            objects = ('N','GA', 'TA', 'IA',
                       'GB', 'TB', 'IB',
                       'GC', 'TC', 'IC')
            y_pos = np.arange(len(objects))
            performance = distri[labels[i]]
            print(type(performance))
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Frequency')
            plt.title('Frequency:'+self.dictionary[labels[i]])

            plt.show()

    def topic_distribution(self, ratio=0.8):
        dictionary = {'watch_s':0, 'watch_s2':1, 'watch_c':2,
                           'play_s':3, 'play_s2':4, 'play_c':5,
                           'search_s':6, 'search_s2':7, 'search_c':8}
        labels = []
        for label in self.labels:
            labels.append(dictionary[label])
        labels = np.array(labels)

        train_pos = np.random.choice(labels.shape[0], int(labels.shape[0]*ratio), replace=False)
        total_pos = np.arange(0,labels.shape[0])
        test_pos = set(total_pos) - set(train_pos)
        test_pos = np.array(list(test_pos))
        train_labels = labels[train_pos]
        test_labels = labels[test_pos]
        trains = []
        tests = []
        for i in range(9):
            trains.append(len(train_labels[train_labels==i]))
            tests.append(len(test_labels[test_labels==i]))
        trains = np.array(trains)
        tests = np.array(tests)
        distris = [trains, tests]
        titles = ['training set','testing set']
        for i in range(2):
            objects = ['WA','WB','WC','PA','PB','PC','SA','SB','SC']
            y_pos = np.arange(len(objects))
            performance = distris[i]
            print(performance)
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Frequency')
            plt.title('Frequency:'+titles[i])
            plt.show()

    def train_and_testing_loss(self, trainfile='train_loss.p', testfile='test_loss.p'):
        with open(trainfile,'rb') as fin:
            train_loss = pickle.load(fin)
            fin.close()

        with open(testfile,'rb') as fin:
            test_loss = pickle.load(fin)

        print(max(train_loss))
        # plt.plot(train_loss)
        # plt.plot(test_loss)
        # plt.legend(['training set','testing set'])
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('GloVe Accuracy')
        # plt.show()

    def plot_group_bar(self, actual, learned, label, groups=9):
        n_groups = groups
        print(actual.shape)
        print(learned.shape)
        actual = actual/float(np.sum(actual))
        means_frank = actual
        means_guido = learned
        # objects = ['WA', 'WB', 'WC', 'PA', 'PB', 'PC', 'SA', 'SB', 'SC']
        objects = ('N', 'GA', 'TA', 'IA',
                   'GB', 'TB', 'IB',
                   'GC', 'TC', 'IC')
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_frank, bar_width,
                         alpha=opacity,
                         color='b',
                         label='actual')

        rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                         alpha=opacity,
                         color='g',
                         label='learned')

        plt.xlabel('words')
        plt.ylabel('Frequency')
        plt.title('Naive Bayes:topic '+label)
        plt.xticks(index + bar_width, set(objects))
        plt.legend()

        plt.tight_layout()
        plt.show()

    def bayes_prior(self, prior, feature, label, ratio=0.8):
        with open(prior, 'rb') as fin:
            prior_pro = pickle.load(fin)
        with open(feature, 'rb') as fin:
            feature_pro = pickle.load(fin)

        dictionary = {'watch_s': 0, 'watch_s2': 1, 'watch_c': 2,
                      'play_s': 3, 'play_s2': 4, 'play_c': 5,
                      'search_s': 6, 'search_s2': 7, 'search_c': 8}
        labels = []
        for label in self.labels:
            labels.append(dictionary[label])
        labels = np.array(labels)

        train_pos = np.random.choice(labels.shape[0], int(labels.shape[0] * ratio), replace=False)
        train_labels = labels[train_pos]
        trains = []

        for i in range(9):
            trains.append(len(train_labels[train_labels == i]))
        trains = np.array(trains)

        self.plot_group_bar(trains, prior_pro, label)

    def bayes_feature(self, feature, ratio=0.8):
        with open(feature, 'rb') as fin:
            feature_pro = pickle.load(fin)

        dictionary = {'watch_s': 0, 'watch_s2': 1, 'watch_c': 2,
                      'play_s': 3, 'play_s2': 4, 'play_c': 5,
                      'search_s': 6, 'search_s2': 7, 'search_c': 8}
        distri = dict()
        for i in range(self.bows.shape[0]):
            label = self.labels[i]
            if not label in distri.keys():
                distri[label] = self.bows[i]
            else:
                distri[label] += self.bows[i]
        labels = distri.keys()
        for i in range(len(labels)):
            objects = ('N', 'GA', 'TA', 'IA',
                       'GB', 'TB', 'IB',
                       'GC', 'TC', 'IC')
            y_pos = np.arange(len(objects))
            performance = distri[labels[i]]

            self.plot_group_bar(performance, feature_pro[dictionary[labels[i]]], labels[i], groups=10)

    def noise_loss(self):
        mags = [0,1, 3,5]

        for mag in mags:
            if mag == 0:
                trainfile = 'test_loss.p'
            else:
                trainfile = 'test_loss_'+str(mag)+'.p'
            with open(trainfile,'rb') as fin:
                train_loss = pickle.load(fin)
            iterations = range(len(train_loss))
            plt.plot(iterations, train_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy with Noisy Data')
        plt.legend(['mag=0', 'mag=1', 'mag=3', 'mag=5'])
        plt.show()

if __name__ == "__main__":
    visualizer = Visualizer('bow_features.p','labels.p')
    visualizer.noise_loss()