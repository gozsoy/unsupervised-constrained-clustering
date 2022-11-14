import numpy as np
import tensorflow as tf
import random
from scipy.sparse import csr_matrix
import scipy.sparse
import itertools

from transformation import preprocess_for_train


def csr_matrix_indices(S):
    """
    Return a list of the indices of nonzero entries of a csr_matrix S
    """
    major_dim, minor_dim = S.shape
    minor_indices = S.indices

    major_indices = np.empty(len(minor_indices), dtype=S.indices.dtype)
    scipy.sparse._sparsetools.expandptr(major_dim, S.indptr, major_indices)

    return major_indices, minor_indices

# borrowed from DC-GMM, added on-the-fly feature extraction for task 1.
class DataGenerator():
    'Generates data for Keras'

    def __init__(self, X, Y, alpha=1000, batch_size=100, num_constrains=0, q=0, ml=0, shuffle=True, l=0, feature_extractor=None):
        'Initialization'
        self.batch_size = batch_size
        self.alpha = alpha
        self.q = q
        self.num_constrains = num_constrains
        self.ml = ml
        self.X = X
        if l == 0:
            self.l = len(Y)
        else:
            self.l = l
        self.Y = Y
        self.W, self.ml_ind1, self.ml_ind2, self.cl_ind1, self.cl_ind2 = self.get_W()
        self.ind1 = np.concatenate([self.ml_ind1,self.cl_ind1])
        self.ind2 = np.concatenate([self.ml_ind2,self.cl_ind2])
        self.indexes = np.arange(len(self.Y))
        self.ind_constr = np.arange(self.num_constrains)
        self.shuffle = shuffle
        self.feature_extractor = feature_extractor

    def transitive_closure(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
        """
        This function calculate the total transtive closure for must-links and the full entailment
        for cannot-links.

        # Arguments
            ml_ind1, ml_ind2 = instances within a pair of must-link constraints
            cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
            n = total training instance number
        # Return
            transtive closure (must-links)
            entailment of cannot-links
        """
        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in zip(ml_ind1, ml_ind2):
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in zip(cl_ind1, cl_ind2):
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)
        ml_res_set = set()
        cl_res_set = set()
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))
                if i <= j:
                    ml_res_set.add((i, j))
                else:
                    ml_res_set.add((j, i))
        for i in cl_graph:
            for j in cl_graph[i]:
                if i <= j:
                    cl_res_set.add((i, j))
                else:
                    cl_res_set.add((j, i))
        ml_res1, ml_res2 = [], []
        cl_res1, cl_res2 = [], []
        for (x, y) in ml_res_set:
            ml_res1.append(x)
            ml_res2.append(y)
        for (x, y) in cl_res_set:
            cl_res1.append(x)
            cl_res2.append(y)
        return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)

    def generate_random_pair(self, y, num, q):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, self.l - 1)
            tmp2 = random.randint(0, self.l - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                if ii >= q:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
                else:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
            else:
                if ii >= q:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
                else:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
            num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_ml(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_cl(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] != y[tmp2]:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def get_W(self):
        if self.ml==0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair(self.Y, self.num_constrains, self.q)
            if self.q == 0:
                ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, self.X.shape[0])
        elif self.ml == 1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_ml(self.Y, self.num_constrains)
        elif self.ml == -1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_cl(self.Y, self.num_constrains)
        print("\nNumber of ml constraints: %d, cl constraints: %d.\n " % (len(ml_ind1), len(cl_ind1)))
        
        ind1 = np.concatenate([ml_ind1, ml_ind2, cl_ind1, cl_ind2])
        ind2 = np.concatenate([ml_ind2, ml_ind1, cl_ind2, cl_ind1])
        data = np.concatenate([np.ones(len(ml_ind1)*2), np.ones(len(cl_ind1)*2)*-1])
        W = csr_matrix((data, (ind1, ind2)), shape=(len(self.X), len(self.X)))
        W = W.tanh().rint()
        return W, ml_ind1, ml_ind2, cl_ind1, cl_ind2


    def extract_image_features(self, batch_imgs):
        
        X = self.feature_extractor(batch_imgs, training=False)

        return X


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    
    def gen(self):
        while True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.ind_constr)
            for index in range(int(len(self.X)/ self.batch_size)):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                X = tf.gather(self.X,indexes)
                
                # if we have feature extractor, this means we are in task 1, othw task 2
                if self.feature_extractor is not None:
                    X = self.extract_image_features(X)
                
                Y = self.Y[indexes]
                W = self.W[indexes][:, indexes]* self.alpha
                ind1, ind2 = csr_matrix_indices(W)
                data = W.data
                
                yield (X, (ind1, ind2, data)), {"output_1": X, "output_4": Y}
            for index in range(self.num_constrains// self.batch_size):
                indexes = self.ind_constr[index * self.batch_size//2:(index + 1) * self.batch_size//2]
                indexes = np.concatenate([self.ind1[indexes], self.ind2[indexes]])
                indexes = indexes.astype(np.int32)

                np.random.shuffle(indexes)
                X = tf.gather(self.X,indexes)

                # if we have feature extractor, this means we are in task 1, othw task 2
                if self.feature_extractor is not None:
                    X = self.extract_image_features(X)
                
                Y = self.Y[indexes]
                W = self.W[indexes][:, indexes]* self.alpha
                ind1, ind2 = csr_matrix_indices(W)
                data = W.data
                #W = W.toarray()
                yield (X, (ind1,ind2, data)), {"output_1": X, "output_4": Y}


# our contribution.
class UnsupervisedDataGenerator():
    'random cannot links, must links from augmentations'

    def __init__(self, X, Y, alpha=1000, batch_size=100, num_constrains=0, q=0, ml=0, shuffle=True, l=0, feature_extractor=None):
        'Initialization'
        self.batch_size = batch_size
        self.alpha = alpha
        self.q = q
        self.num_constrains = num_constrains
        self.ml = ml
        self.X = X
        if l == 0:
            self.l = len(Y)
        else:
            self.l = l
        self.Y = Y

        self.ml_ind1, self.ml_ind2, self.cl_ind1, self.cl_ind2 = self.get_W()
        self.ind1 = np.concatenate([self.ml_ind1,self.cl_ind1])
        self.ind2 = np.concatenate([self.ml_ind2,self.cl_ind2])
        self.indexes = np.arange(len(self.Y))
        self.ind_constr = np.arange(self.num_constrains)
        self.shuffle = shuffle
        self.feature_extractor = feature_extractor

        self.transformation1 = self.get_transformation_pipeline()
        self.transformation2 = self.get_transformation_pipeline()

        self.ml_link_per_batch = 50


    def get_transformation_pipeline(self):

        if self.X.shape[1] == 96: # STL-10 dataset

            transformation = tf.keras.Sequential([   
                tf.keras.layers.experimental.preprocessing.RandomCrop(height=72, width=72),
                tf.keras.layers.experimental.preprocessing.Resizing(height=96, width=96),
                tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.4)
                ])
            return transformation
        
        elif self.X.shape[1] == 32: # CIFAR-10 dataset
            
            transformation = tf.keras.Sequential([   
                tf.keras.layers.experimental.preprocessing.RandomCrop(height=24, width=24),
                tf.keras.layers.experimental.preprocessing.Resizing(height=32, width=32),
                tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.4)
                ])
            return transformation

        else:
            raise NotImplemented("Invalid dataset {}".format(self.type))


    def transitive_closure(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
        """
        This function calculate the total transtive closure for must-links and the full entailment
        for cannot-links.

        # Arguments
            ml_ind1, ml_ind2 = instances within a pair of must-link constraints
            cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
            n = total training instance number
        # Return
            transtive closure (must-links)
            entailment of cannot-links
        """
        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in zip(ml_ind1, ml_ind2):
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in zip(cl_ind1, cl_ind2):
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)
        ml_res_set = set()
        cl_res_set = set()
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))
                if i <= j:
                    ml_res_set.add((i, j))
                else:
                    ml_res_set.add((j, i))
        for i in cl_graph:
            for j in cl_graph[i]:
                if i <= j:
                    cl_res_set.add((i, j))
                else:
                    cl_res_set.add((j, i))
        ml_res1, ml_res2 = [], []
        cl_res1, cl_res2 = [], []
        for (x, y) in ml_res_set:
            ml_res1.append(x)
            ml_res2.append(y)
        for (x, y) in cl_res_set:
            cl_res1.append(x)
            cl_res2.append(y)
        return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)

    def generate_random_pair(self, y, num, q):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, self.l - 1)
            tmp2 = random.randint(0, self.l - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                if ii >= q:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
                else:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
            else:
                if ii >= q:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
                else:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
            num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_ml(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_cl(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] != y[tmp2]:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def get_W(self):
        if self.ml==0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair(self.Y, self.num_constrains, self.q)
            if self.q == 0:
                ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, self.X.shape[0])
        elif self.ml == 1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_ml(self.Y, self.num_constrains)
        elif self.ml == -1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_cl(self.Y, self.num_constrains)
        '''print("\nNumber of ml constraints: %d, cl constraints: %d.\n " % (len(ml_ind1), len(cl_ind1)))
        
        ind1 = np.concatenate([ml_ind1, ml_ind2, cl_ind1, cl_ind2])
        ind2 = np.concatenate([ml_ind2, ml_ind1, cl_ind2, cl_ind1])
        data = np.concatenate([np.ones(len(ml_ind1)*2), np.ones(len(cl_ind1)*2)*-1])
        W = csr_matrix((data, (ind1, ind2)), shape=(len(self.X), len(self.X)))
        W = W.tanh().rint()'''
        return ml_ind1, ml_ind2, cl_ind1, cl_ind2


    def extract_image_features(self, batch_imgs):

        X = self.feature_extractor(batch_imgs, training=False)

        return X


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    
    def gen(self):
        while True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.ind_constr)
            for index in range(int(len(self.X)/ self.batch_size)):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                X = tf.gather(self.X,indexes)

                # if we have feature extractor, this means we are in task 1, othw task 2
                if self.feature_extractor is not None:
                    X = self.extract_image_features(X)
                
                Y = self.Y[indexes]

                ind1 = np.array([],dtype=np.int32)
                ind2 = np.array([],dtype=np.int32)
                data = np.array([],dtype=np.float64)
                
                yield (X, (ind1, ind2, data)), {"output_1": X, "output_4": Y}

            for index in range(self.num_constrains// self.batch_size):
                indexes = self.ind_constr[index * self.batch_size//2:(index + 1) * self.batch_size//2]
                indexes = np.concatenate([self.ind1[indexes], self.ind2[indexes]]).astype(int)

                np.random.shuffle(indexes)

                X = self.X[indexes]
                X_aug1 = self.transformation1(X, training=True)
                X_aug2 = self.transformation2(X, training=True)
                X = tf.concat((X_aug1, X_aug2[:self.ml_link_per_batch]),axis=0)

                # if we have feature extractor, this means we are in task 1, othw task 2
                if self.feature_extractor is not None:
                    X = self.extract_image_features(X)

                # Y is only for measuring performance metrics
                Y = tf.concat((self.Y[indexes],self.Y[indexes][:self.ml_link_per_batch]),axis=0)

                # create random cannot link constraints
                ind1 = np.arange(0,self.batch_size)
                ind2 = (np.zeros(self.batch_size) - 1).astype(int)
                data = np.array([-self.alpha] * self.batch_size)

                array2sample = np.arange(0,self.batch_size)
                for i in range(len(ind2)):

                    if ind2[i]!=-1:
                        continue
                    else:
                        sampled_idx = i
                        
                        while sampled_idx == i or ind2[sampled_idx]!=-1:
                            sampled_idx = np.random.choice(array2sample)

                        array2sample = array2sample[(array2sample != sampled_idx) 
                                                    & (array2sample != i)]

                        ind2[i] = sampled_idx
                        ind2[sampled_idx] = i
                
                # add must link constraints created via augmentation
                ind1 = np.append(ind1, np.arange(0,self.ml_link_per_batch))
                ind2 = np.append(ind2, np.arange(self.batch_size, self.batch_size + self.ml_link_per_batch))
                data = np.append(data, np.array([self.alpha] * self.ml_link_per_batch))

                yield (X, (ind1,ind2, data)), {"output_1": X, "output_4": Y}


# our contribution. we use it for contrastive learning pretraining
class ContrastiveDataGeneratorPretraining():
    'Generates data within contrastive learning schema for AE'

    def __init__(self, X, batch_size=64):
        'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.indexes = np.arange(len(self.X))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    
    def gen(self):
        while True:
            np.random.shuffle(self.indexes)
            for index in range(int(len(self.X)/ self.batch_size)):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

                X = tf.gather(self.X,indexes)

                # concatenate 
                X = tf.concat((X, X),axis=0)  # (N,96,96,3) -> (2N,96,96,3)

                # augment
                X = tf.map_fn(preprocess_for_train,tf.cast(X,dtype=tf.float32))
                
                # format only suitable for pretraining
                yield X, {"output_1": X}