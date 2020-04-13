import numpy as np

def poly(x, p=2):
    x = np.array(x)
    X = np.transpose(np.vstack([x**k for k in range(p+1)]))
    return np.linalg.qr(X)[0][:,1:]


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)