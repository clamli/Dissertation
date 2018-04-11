import numpy as np

def k_medoids(distance_matrix, K, max_iterations=20):
    user_num = distance_matrix.shape[0]
    random_index = np.random.permutation(user_num)
    centroids = np.sort(random_index[0:K])
    for time in range(max_iterations):
        # Assign user to clusters
        print("K-medoids: %d / %d"%(time+1, max_iterations), end="\r")
        user_centroids_dist = distance_matrix[:, centroids]
        indices = np.argmin(user_centroids_dist, axis=1)
        indices = centroids[indices]
        indices[centroids] = centroids
        centroids_update = np.zeros(K, 'int32')
        # Find new medoids
        for i in range(K):
            cluster_list = (indices == centroids[i])
            cluster = distance_matrix[np.ix_(cluster_list, cluster_list)]
            cluster_list_ = np.where(cluster_list != 0)[0]
            new_center = np.argmin(np.sum(cluster, axis=1), axis=0)
            centroids_update[i] = cluster_list_[new_center]
        centroids_update = np.sort(centroids_update)
        # Termination condition
        if (centroids == centroids_update).all():
            print('Iteration stop')
            break
        else:
            centroids = centroids_update
    # Assign user to clusters
    user_cluster_set = []
    for i in range(K):
        user_cluster_set.append(list(np.where((indices==centroids[i]) != 0)[0]))
    return user_cluster_set