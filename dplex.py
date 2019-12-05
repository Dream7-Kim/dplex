import jax.numpy as np
import logging

class dplex:
    def __init__(self, a, b):
        if(a.shape != b.shape):
            logging.warning('The shape of real and imag is different')
        else:
            self.val = np.stack([a, b], axis=0)

    def __repr__(self):
        return 'dream7.dplex: ' + np.array_repr(self.val[0] + self.val[1]*(1j))

    def __add__(self, rhs):
        if(self.val[0].shape != rhs.val[0].shape):
            logging.warning('The shape of real and imag is different in + operation')
        else:
            return dplex(self.val[0] + rhs.val[0], self.val[1] + rhs.val[1])

    def __sub__(self, rhs):
        if(self.val[0].shape != rhs.val[0].shape):
            logging.warning('The shape of real and imag is different in + operation')
        else:
            return dplex(self.val[0] - rhs.val[0], self.val[1] - rhs.val[1])

    def __mul__(self, rhs):
        if(self.val[0].shape != rhs.val[0].shape):
            logging.warning('The shape of real and imag is different in + operation')
        else:
            return dplex(self.val[0] * rhs.val[0] - self.val[1] * rhs.val[1],
                self.val[0] * rhs.val[1] + self.val[1] * rhs.val[0])

    def __truediv__(self, rhs):
        if(self.val[0].shape != rhs.val[0].shape):
            logging.warning('The shape of real and imag is different in + operation')
        else:
            temp = self * dconj(rhs)
            temp1 = dabs(rhs) ** 2
            return dplex(temp.val[0] / temp1, temp.val[1] / temp1)

    def __rtruediv__(self, lhs):
        temp = dconj(self)
        temp1 = dabs(self)
        return dplex(lhs * temp.val[0] / temp1, lhs * temp.val[1] / temp1)


def deinsum(subscript, aa, bb):
    real = np.einsum(subscript, aa.val[0], bb.val[0]) - np.einsum(subscript, aa.val[1], bb.val[1])
    imag = np.einsum(subscript, aa.val[0], bb.val[1]) + np.einsum(subscript, aa.val[1], bb.val[0])
    return dplex(real, imag)

def dabs(aa):
    return aa.val[0]**2 + aa.val[1]**2

def dconj(aa):
    return dplex(aa.val[0], -aa.val[1])


