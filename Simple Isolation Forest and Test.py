"""
Created on Fri Sep 27 15:52:22 2019

@author: Yubo
"""
import numpy as np
import pandas as pd


def c_of_n(n):
    c = 2*(np.log(n-1) + 0.5772) - 2*(n-1)/n
    return c

class InternalTreeNode:
    '''
    Internal Node of a tree
    Attr:
        split_attr: The attribute used to split at this node
        split_value: The value of attribute used to split
        left_node: left of node
        right_node: right of node
    Method:
        self.isExNode(): To decide whether is a leef node
    '''
    def __init__(self, split_attr, split_value, left_node, right_node):
        self.split_attr = split_attr
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
    def isExNode(self):
        return False
 
        
        
class ExNode:
    def __init__(self, size):
        self.size = size
    def isExNode(self):
        return True
  

def choose_data(data, split_attr):
    return data[:, split_attr]

def subsampling(data, subsampling_ratio = 0.05):
    """
    Input:
        data: data to subsample
        subsampling_ratio(0.05): subsample to data
    Output:
        subdata: data with size of subsampling size
        
    Isolation Tree works well when subsample. And it can increase speed.
    
    """
    subsampling_size = int(subsampling_ratio*len(data))
    indices = np.random.permutation(len(data))
    subsampling_indices = indices[:subsampling_size]
    return data[subsampling_indices]

      
def split(data, split_attr, split_value):
    data_attr = choose_data(data, split_attr)
    data_l = data[data_attr < split_value]
    data_r = data[data_attr >= split_value]
    return data_l, data_r
    
def random_select_attr_value(data):
    dimension = data.shape[1]
    attr = np.random.choice(dimension) # No data abstraction
    data = choose_data(data, attr)
    minimun, maximum = data.min(), data.max()
    split_value = np.random.uniform(minimun, maximum)
    return attr, split_value


def iTree(data, e, l):
    '''
    Input:
        data: The data to create isolation tree
        e: Current Tree Height
        l: Height Limit
    '''
    if len(data) <= 1 or e > l:
        return ExNode(size = len(data))
    attr, split_value = random_select_attr_value(data)
    data_l, data_r = split(data, attr, split_value)
    
    return InternalTreeNode(attr, split_value,
                            left_node = iTree(data_l, e+1, l),
                            right_node = iTree(data_r, e+1, l))
    
def iForest(data, t, subsampling_ratio = 0.05):
    '''
    Input:
        data: data
        t: The number of itrees
        subsampling_ratio: ratio of data size to subdata size
    '''
    forest = []
    sub_size = int(len(data)*subsampling_ratio)
    tree_height_limit = np.log2(sub_size) + 1
    for i in range(t):
        sub_data = subsampling(data, subsampling_ratio)
        tree = iTree(sub_data, 0, tree_height_limit)
        forest.append(tree)
    return forest

def path_length(x, itree, e):
    '''
    Input:
        x: An instance
        itree: An itreeNode
        e: current path length
    '''
    if itree.isExNode():
        if itree.size <=2 :
            return e 
        else:
            return e + c_of_n(itree.size-1)
    attr = itree.split_attr
    value = x[attr]
    if value < itree.split_value:
        return path_length(x, itree.left_node, e+1)
    elif value >= itree.split_value:
        return path_length(x, itree.right_node, e+1)

def average_length(x, forest):
    length = np.zeros(len(forest))
    for index, itree in enumerate(forest):
        length[index] = path_length(x, itree, 0)
    return length.mean()

###### Ploting and Testing

def create_test_data(data_size = 5000, dimension = 2,
                     mean_normal = (0,0), std_normal = (10,15),
                     mean_outlier = (0,16), std_outlier = (1,1),
                     outlier_ratio = 0.02):
    
    if len(std_normal) != dimension:
        return None
    outlier_size = int(data_size*outlier_ratio)
    
    sigma_normal = np.diag(std_normal)
    sigma_outlier = np.diag(std_outlier)
    
    regular_data = np.random.multivariate_normal(mean_normal, sigma_normal, size = data_size)
    regular_class_indicator = np.ones(len(regular_data))[:,np.newaxis]
    regular_data = np.concatenate([regular_data, regular_class_indicator], axis = 1)
    
    outlier_data = np.random.multivariate_normal(mean_outlier, sigma_outlier, size = outlier_size)
    outlier_class_indicator = np.ones(len(outlier_data))[:, np.newaxis] * 2
    outlier_data = np.concatenate([outlier_data, outlier_class_indicator], axis = 1)
    
    data = np.concatenate([regular_data, outlier_data])
    indices = np.random.permutation(len(data))
    data = data[indices]
    
    return data
    

    
if __name__ == '__main__':
    data = create_test_data()
    import matplotlib.pyplot as plt
    plt.figure(figsize = (16,9))
    plt.scatter(data[:,0], data[:,1], c = data[:,2])
    
    sub_data = subsampling(data)
    plt.figure(figsize = (16,9))
    plt.scatter(sub_data[:,0], sub_data[:,1], c = sub_data[:,2])
    
    test_data = data[:,:2]
    iforest = iForest(test_data, 200, 0.05)
    
    test_result = np.zeros((len(test_data),1))
    for i in range(len(test_result)):
        x = test_data[i,:]
        test_result[i] = average_length(x, iforest)
    data = np.concatenate([data, test_result], axis = 1)
    
    data1 = data[data[:,2]==1]
    data2 = data[data[:,2]==2]
    
    data1[:,2] = 20
    data2[:,2] = 100
    
    plt.figure(figsize = (16,9))
    plt.scatter(data[:,0],data[:,1],
                c = data[:,3], cmap = plt.get_cmap('jet'))
    plt.colorbar()
    plt.scatter(data2[:,0],data2[:,1],facecolor='none',edgecolor='b',s=200)
    plt.show()
