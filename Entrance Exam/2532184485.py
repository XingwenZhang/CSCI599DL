# coding: utf-8

import pickle
from sklearn.decomposition import PCA
import numpy as np
import sys


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def data_handle(filepath, N):
    data_dict = unpickle(filepath)
    # The keys of dict are b'batch_label', b'labels', b'data', b'filenames'
    data = data_dict.get(b'data',None)
    labels = data_dict.get(b'labels',None)
    # assert data.any() and labels, "data, labels have something wrong"
    # Fetch first 1000 images and labels
    data = data[:1000,:]
    labels = labels[:1000]
    # print(data.shape)
    data_train, data_test = data[N:,:],data[:N,:]
    labels_train, labels_test = labels[N:],labels[:N]
    
    return data_train,labels_train,data_test,labels_test


def convert_rgb_to_gray(data):
    """Convert rgb image to gray image
    
    Each row of the array stores a 32x32 colour image. 
    The first 1024 entries contain the red channel values, 
    the next 1024 the green, and the final 1024 the blue. 
    The image is stored in row-major order, so that the first 
    32 entries of the array are the red channel values of 
    the first row of the image.
    
    Formula: L(Grayscale) = 0.299R + 0.587G + 0.114B
    
    Args:
        data : rgb image data, shape=(rows*3072)
        
    Returns:
        gray_image: gray image data, shape=(rows*1024)
    """

    gray_image = 0.299 * data[:,:1024] + 0.587 * data[:,1024:2048] + 0.114 * data[:,2048:]
    return gray_image


def pca_handle(data_train, data_test, D):
    data_gray_train = convert_rgb_to_gray(data_train)
    data_gray_test = convert_rgb_to_gray(data_test)
    pca = PCA(n_components=D, svd_solver='full')
    pca.fit(data_gray_train)
    data_pca_train = pca.transform(data_gray_train)
    data_pca_test = pca.transform(data_gray_test)
    return data_pca_train, data_pca_test


def knn(K, data_train, labels_train, data_test):
    """Brute-Force KNN algorithm, metric is inverse Euclidean distance
    """
    neighbors = np.zeros(10)
    distance = np.linalg.norm(data_train-data_test,axis=1)
    K_smallest_index = np.argsort(distance)[:K] # Not an efficient choice
    # labels_array = np.asarray(labels_train)
    # neighbors[labels_array[K_smallest_index]] = 1./distance[K_smallest_index]
    for index in K_smallest_index:
        neighbors[labels_train[index]] += 1./distance[index]
    
    label_test = np.argsort(neighbors)[-1]
    return label_test


def main():
    assert len(sys.argv) == 5, 'parameters miss'
    K = int(sys.argv[1])
    D = int(sys.argv[2])
    N = int(sys.argv[3])
    file_path = sys.argv[4]
    
    image_train, labels_train, image_test, labels_test = data_handle(file_path, N)
    
    image_pca_train, image_pca_test = pca_handle(image_train, image_test, D)
    
    rows,_ = image_pca_test.shape
    labels_predict = []
    for row in range(rows):
        labels_predict.append(knn(K, image_pca_train, labels_train, image_pca_test[row, :]))
    
    with open('./2532184485.txt','w') as f:
        for predict_label, truth_label in zip(labels_predict, labels_test):
            f.write(str(predict_label) + ' ' + str(truth_label) + '\n')
    print('Finished')

if __name__ == "__main__":
    main()

