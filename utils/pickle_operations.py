import pickle


def save(path, content):
    """
    Save python object into given path
    :param content:
    :param path:
    :return:
    """
    f = open(path, 'wb')
    pickle.dump(content, f)
    f.close()


def load(path):
    """
    Reconstruct python object from file
    :param path:
    :return:
    """
    f = open(path, 'rb')
    content = pickle.load(f)
    f.close()
    return content