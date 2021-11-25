"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/1/2019
"""


def optimize_model(model=None, data=(), limit=(1, 100, 1), gamma=None, normalize=False, verbose=False):
    """
    :param model:
    :param data:
    :param limit:
    :param gamma:
    :param normalize:
    :param verbose:
    :return:
    """
    assert len(data) == 4
    assert len(limit) == 3
    assert limit[0] <= 0 or limit[0] <= limit[1]
    if model is None:
        return None
    accuracy = []
    for c in range(limit[0], limit[1], limit[2]):
        model.fit_parameter(c=c, kernel='rbf', gamma=1e-04)
        acc = model.classify(data[0], data[1], data[2], data[3], normalize=normalize)
        if verbose is True:
            print("for c = {}, Accuracy : {} . ".format(c, acc))
        accuracy.append((c, acc))
    return accuracy
