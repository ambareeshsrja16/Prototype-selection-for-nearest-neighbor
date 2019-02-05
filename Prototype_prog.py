import numpy as np


class Prototype_Selector:
    def __init__(self, data_x, datalabel_y, M = 10):
        """

        :param data_x:
        :param datalabel_y:
        :param test_x:
        :param testlabel_y:
        """
        self.x_train = np.array(data_x)
        self.y_train = np.array(datalabel_y)
        self.M = M

        self.bbag = None
        self.gratios = None

    def k_mean_cluster_return_centroid(self, population, centroid_number=10):
        """
        returns centroids from population
        """
        from sklearn.cluster import KMeans

        original_shape = np.array(population).shape
        X = np.array(population).reshape(-1, np.size(population[0]))
        kmeans = KMeans(n_clusters=centroid_number, random_state=0).fit(X)
        # kmeans = KMeans(n_clusters=centroid_number).fit(X)
        centroids = kmeans.cluster_centers_
        needed_shape = tuple([len(centroids)]+[i for i in original_shape[1:]])
        centroids = centroids.reshape(needed_shape)
        print("Centroids ", centroids.shape)

        return centroids

    def get_label_ratios(self, labels):
        """
        returns dictionary of labels with ratios
        """
        from collections import Counter
        b = Counter(labels)
        ratio_labels_dict = {i: j / len(labels) for i, j in b.items()}

        return ratio_labels_dict

    def bag_of_elements_label(self, labels):
        """
        returns the dict_of_labels_to_indexes

        { 1: [],
          2: []...
        }

        Where:
        1 : array([[    3],
           [    6],
           [    8],
           ...,
           [59979],
           [59984],
           [59994]]

        For MNIST:

        bag_of_elements_label(trainingLabels).keys():
        dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        """
        unique_labels = set(labels)
        labels = np.array(labels)
        dict_of_labels_to_indexes = {i: np.argwhere(labels == i).flatten() for i in unique_labels}

        return dict_of_labels_to_indexes

    def get_number_of_centroids_required(self, prototype_size):
        """
        given prototype size and ratio of labels return actual number of elements needed
        """
        import math
        import random
        actual_ratios = {i: math.ceil(prototype_size * self.gratios[i]) for i in self.gratios}

        while sum(actual_ratios.values()) != prototype_size:
            actual_ratios[random.randint(0, len(actual_ratios) - 1)] -= 1

        return actual_ratios

    def prototype_generator(self):
        """

        :return: prototype_generated, prototype_generated_labels
        """

        self.gratios = self.get_label_ratios(self.y_train)
        self.bbag = self.bag_of_elements_label(self.y_train)
        actual_ratios = self.get_number_of_centroids_required(prototype_size=self.M)

        needed_shape = tuple([self.M] + [i for i in self.x_train.shape[1:]])
        prototype_generated = np.zeros(needed_shape)
        prototype_generated_labels = np.zeros(self.M, )

        count = 0
        for label, ratio_req in actual_ratios.items():
            population = self.x_train[self.bbag[label]]
            temp = self.k_mean_cluster_return_centroid(population, ratio_req)
            prototype_generated[count:count + len(temp)] = temp
            prototype_generated_labels[count:count + len(temp)] = [label] * len(temp)
            count += len(temp)

        return prototype_generated, prototype_generated_labels


def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte',
                         dtype='ubyte')[2 * intType.itemsize:]

    return data, labels


def do_KNN_classification(X_train, Y_train, X_test, Y_test, k=1):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics

    X_train = np.array(X_train).reshape(len(X_train), -1)
    X_test = np.array(X_test).reshape(len(X_test), -1)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, Y_train)
    Y_pred = neigh.predict(X_test)

    test_acc = metrics.accuracy_score(Y_test, Y_pred)
    print(test_acc)
    return test_acc * 100


def pick_random_elements_from_Training_set(X_train, Y_train, random_elem_size=100):
    random_indices = np.random.choice(X_train.shape[0], replace=False, size=random_elem_size)
    X_random = X_train[random_indices]
    Y_random = Y_train[random_indices]

    return X_random, Y_random


def plotter_function_error_bars(X_axis_vals, Y_axis_vals, Y_std_dev_vals):
    import matplotlib.pyplot as plt
    x, y, std_dev = X_axis_vals, Y_axis_vals, Y_std_dev_vals

    plt.figure()
    plt.errorbar(x, y, yerr=std_dev, fmt='--', mfc='red')
    plt.title("Accuracy across Prototype Size M ")

    # plotted with std Dev Error, Each experiment was iterated 100 times "


def mean_accuracy_std_dev_over_iterations(X_train, Y_train, X_test, Y_test, M=100, iteration_times=5):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_Test:
    :param M:
    :param iteration_times:
    :return:
    """

    mean_acc_prototyper = []
    mean_acc_random_selector = []

    for i in range(iteration_times):
        inst = Prototype_Selector(X_train, Y_train, M)
        proto_train, proto_label = inst.prototype_generator()

        X_r, Y_r = pick_random_elements_from_Training_set(X_train, Y_train, random_elem_size=M)
        mean_acc_prototyper.append(do_KNN_classification(proto_train, proto_label, X_test, Y_test))
        mean_acc_random_selector.append(do_KNN_classification(X_r, Y_r, X_test, Y_test))

    mean_acc_val_P = np.average(mean_acc_prototyper)
    std_dev_acc_val_P = np.std(mean_acc_prototyper)

    mean_acc_val_R = np.average(mean_acc_random_selector)
    std_dev_acc_val_R = np.std(mean_acc_random_selector)

    return mean_acc_val_P, std_dev_acc_val_P,mean_acc_val_R, std_dev_acc_val_R


if __name__ == '__main__':
    trainingImages, trainingLabels = loadMNIST("train", "./data")
    testImages, testLabels = loadMNIST("t10k", "./data")

    M = 100  #1000
    mean_acc_val_P, std_dev_acc_val_P,mean_acc_val_R, std_dev_acc_val_R = mean_accuracy_std_dev_over_iterations(trainingImages,trainingLabels,testImages, testLabels, M)
    print(f"\n mean_acc_val_P : {mean_acc_val_P} \n  std_dev_a cc_val_P: {std_dev_acc_val_P} \n  mean_acc_val_R: {mean_acc_val_R} \n, std_dev_acc_val_R: {std_dev_acc_val_R}")