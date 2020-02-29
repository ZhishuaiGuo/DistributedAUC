from random import Random
from parameters import para
import numpy as np

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
         return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    # same seed on all processes, thus the partitions are the same
    def __init__(self, data, sizes, seed=123, neg_keep_ratio=1):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = list(range(data_len))
        #print(data[0][1])
        #print(data[1][1])
        #for i in range(100):
        #    print(data[i][1])
        #exit()
        
        # pos_index = list()
        # neg_index = list()
        ''' 
        neg_index = (data[:][1] <= para.split_index) 
        pos_indes = (data[:][1] > para.split_index)

        neg_index = np.where(data[:][1] <= para.split_index, indexes, -1)
        neg_index = np.index[neg_index!=-1]
        pos_index = np.where(data[:][1] > para.split_index, indexes, 1)
        pos_index = np.index[pos_index!=1]
        '''
        neg_index =  list(np.arange(642289))
        pos_index = list(np.arange(642290, 1281167))
        print("data_len: " + str(data_len))
        '''
        for i in range(0, 10):
            print("i:" +str(i))
            #if data[i][1] > para.split_index:
            #    print("i" + str(i) + "index: " + str(para.split_index))
        count = 0
        current_label = 0 
        for i in range(0, data_len):
            if data[i][1] != current_label:
                print("count " + str(current_label) + "is: " +str(count))
                count = 0
                current_label = data[i][1]
                print("i" + str(i))
                #exit()
            count += 1
        exit()
        '''
        
        rng.shuffle(pos_index)
        rng.shuffle(neg_index) 
        
        neg_num = int(len(neg_index)*neg_keep_ratio)
        #print(neg_num)
        #exit()
        neg_index = neg_index[0:neg_num]
        indexes = pos_index + neg_index
        rng.shuffle(indexes)
       
 
        #indexes = [x for x in range(0, data_len)]
        #print(indexes[0:100])
        # rng.shuffle(indexes)
        #print(indexes[0:100])
        # exit()
        
        
        data_len = len(indexes)        
 		
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
        
