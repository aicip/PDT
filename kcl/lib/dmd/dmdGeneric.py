from abc import abstractmethod, abstractproperty


def fit_checker(func):
    """
    Make sure the DMD operator has been fit before calling the function
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_fit:
            raise RuntimeError("Must call fit first")
        return func(self, *args, **kwargs)

    return wrapper


class TorchDMDGeneric:
    def __init__(self, rank=0):
        self.rank = rank
        self.is_fit = False

    @abstractmethod
    def fit(self, x):
        """
        Fit the DMD model to the data in x
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """
        Predict the next state given the current state
        """
        raise NotImplementedError

    @abstractproperty
    def eigs(self):
        """
        Return the eigenvalues of the DMD model
        """
        raise NotImplementedError

    @abstractproperty
    def basis(self):
        """
        Return the basis of the DMD model
        """
        raise NotImplementedError

    @abstractproperty
    def eigenvectors(self):
        """
        Return the eigenvectors of the DMD model
        """
        raise NotImplementedError

    @abstractproperty
    def left_eigs(self):
        """
        Return the low dimensional left eigenvectors
        """
        raise NotImplementedError
