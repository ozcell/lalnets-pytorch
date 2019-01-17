import numpy as np 

def calculate_cl_acc(ground_truth, est, nb_all_clusters, cluster_offset=0, label_correction=False):

    majority = np.zeros(nb_all_clusters)
    population = np.zeros(nb_all_clusters)

    if label_correction:
        est = correct_labels(ground_truth, est)

    for cluster in range(cluster_offset, nb_all_clusters + cluster_offset):
        if np.bincount(ground_truth[est==cluster]).size != 0:
            majority[cluster-cluster_offset] = np.bincount(ground_truth[est==cluster]).max()
            population[cluster-cluster_offset] = np.bincount(ground_truth[est==cluster]).sum()

    cl_acc = majority[majority>0].sum()/population[population>0].sum()

    return cl_acc, population.sum()

def correct_labels(ground_truth, est):

    corrested_est = np.zeros_like(est, dtype='int')

    for cluster in range(est.max()+1):
        if np.bincount(ground_truth[est==cluster]).size != 0:
            true_label = np.bincount(ground_truth[est==cluster]).argmax()
            corrested_est[est==cluster] = true_label

    return corrested_est

def to_numpy(x):
    return x.detach().cpu().numpy()