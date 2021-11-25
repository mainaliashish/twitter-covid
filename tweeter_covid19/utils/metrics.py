"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/15/2019
"""
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standard_normalize(train_vector=None, vectors=None):
    """
    This function is used to normalize the data.
    :param train_vector: to fit the standard scaler.
    :param vectors: list -> eg. [vector_1, vector_2, ... , vector n]
    :return: normalized vectors
    """
    if train_vector is None or vectors is None:
        return None
    scale_model = StandardScaler()
    scale_model.fit(train_vector)
    _vectors = []
    for vector in vectors:
        _vectors.append(scale_model.transform(vector))
    return _vectors


def reduce_dimension(train_vectors: object = None, test_vectors: object = None, components: object = None, reduce_dim: object = True) -> object:
    """
    :param train_vectors:
    :param test_vectors:
    :param components:
    :param reduce_dim:
    :return:
    """
    assert type(components) == int
    if reduce_dim is False:
        return train_vectors, test_vectors
    pca = PCA(n_components=components)
    scale = preprocessing.StandardScaler().fit(train_vectors)
    scaled_train_vector = scale.transform(train_vectors)
    scaled_test_vector = scale.transform(test_vectors)
    pca.fit(scaled_train_vector)  # fit the model
    train_reduced_vectors = pca.transform(scaled_train_vector)
    test_reduced_vectors = pca.transform(scaled_test_vector)
    return train_reduced_vectors, test_reduced_vectors
