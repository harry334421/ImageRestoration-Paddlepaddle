import numpy as np
import warnings
import os
import paddle
from paddle.framework import dtype
import sklearn.neighbors as neighbors

# import torch
from skimage import color
from paddle.autograd import PyLayer
import pdb

# from paddle.autograd import Function
# ************************
# ***** CAFFE LAYERS *****
# ************************


class NNEncode:
    """Encode points using NearestNeighbors search and Gaussian kernel"""

    def __init__(self, NN, sigma, cc=-1):
        self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        # self.nbrs = neighbors.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        """
        input feature map flatten to [N*H*W, 2]，then find the nerest nn for ab in 313 bins. return feature map [N, 313, H, W] value
        """
        # pts_flt = flatten_nd_array(pts_nd,axis=axis)
        pts_flt = paddle.reshape(paddle.transpose(pts_nd, (0, 2, 3, 1)), (-1, 1, 2))
        pts_flt = paddle.tile(pts_flt, (1, self.K, 1))
        # pts_flt ---> [N*H*W, 2]
        P = pts_flt.shape[0]

        assert self.K == 313
        dist = paddle.sum(paddle.square(pts_flt - self.cc), axis=2)
        dists, inds = paddle.topk(dist, k=self.NN, axis=1, largest=False)
        inds_onehot = paddle.nn.functional.one_hot(inds, self.K)

        wts = paddle.exp(-paddle.square(dists) / (2 * self.sigma**2))
        wts = wts / paddle.unsqueeze(paddle.sum(wts, axis=1), 1)
        wts = paddle.unsqueeze(wts, 2)
        # wts.shape ---> [N*H*W, NN]
        # pdb.set_trace()
        assert wts.shape == [P, self.NN, 1]
        pts_enc_flt = inds_onehot * wts
        pts_enc_flt = paddle.sum(pts_enc_flt, axis=1)
        # (N*H*W, 313)

        return pts_enc_flt


class NNEncLayer(object):
    """Layer which encodes ab map into Q colors,ab_enc
    INPUTS
        bottom[0]   Nx2xXxY
    OUTPUTS
        top[0].data     NxQ
    """

    def __init__(self, cc):
        warnings.filterwarnings("ignore")

        self.NN = 5
        self.sigma = 5.0
        self.ENC_DIR = "model/pretrain/"
        self.nnenc = NNEncode(self.NN, self.sigma, cc)

    def __call__(self, bottom):
        # if len(bottom) == 0:
        #     raise Exception("NNEncLayer should have inputs")
        # print(bottom.shape)
        return self.nnenc.encode_points_mtx_nd(bottom, axis=1)


class PriorBoostLayer(object):
    """Layer boosts ab values based on their rarity
    INPUTS
        bottom[0]       NxQxXxY
    OUTPUTS
        top[0].data     Nx1xXxY
    """

    def __init__(self):

        self.ENC_DIR = "./models/custom_layers/"
        self.gamma = 0.5
        self.alpha = 1.0
        self.pc = PriorFactor(
            self.alpha,
            gamma=self.gamma,
            priorFile=os.path.join(self.ENC_DIR, "prior_probs.npy"),
        )

    def __call__(self, bottom):
        if len(bottom) == 0:
            raise Exception("PriorBoostLayer should have inputs")
        return self.pc.forward(bottom, axis=1)


class NonGrayMaskLayer(object):
    """Layer outputs a mask based on if the image is grayscale or not
    INPUTS
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    """

    def __init__(self):
        self.thresh = 5  # threshold on ab value

    def __call__(self, bottom):
        if len(bottom) == 0:
            raise Exception("NonGrayMaskLayer should have inputs")
        # if an image has any (a,b) value which exceeds threshold, output 1
        return (
            np.sum(np.sum(np.sum(np.abs(bottom) > self.thresh, axis=1), axis=1), axis=1)
            > 0
        )[:, na(), na(), na()]


class Rebalance_Op(PyLayer):
    @staticmethod
    def forward(ctx, input, factors):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input, factors)

        # return tensor * constant
        return input * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        input, factors = ctx.saved_tensor()
        grad_input = grad_factors = None
        # if ctx.needs_input_grad[0]:
        # grad_input = grad_output.mm(weight.t())
        grad_input = grad_output * factors
        # grad_input = grad_output
        # if ctx.needs_input_grad[1]:
        #     grad_factors = grad_output.t().mm(input)
        return grad_input, None


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor:
    def __init__(self, alpha, gamma=0, verbose=True, priorFile=""):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.0
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (
            1 - self.gamma
        ) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor / np.sum(
            self.prior_probs * self.prior_factor
        )  # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs * self.prior_factor
        self.implied_prior = self.implied_prior / np.sum(
            self.implied_prior
        )  # re-normalize

        if self.verbose:
            self.print_correction_stats()

    def print_correction_stats(self):
        print("Prior factor correction:")
        print("  (alpha,gamma) = (%.2f, %.2f)" % (self.alpha, self.gamma))
        print(
            "  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)"
            % (
                np.min(self.prior_factor),
                np.max(self.prior_factor),
                np.mean(self.prior_factor),
                np.median(self.prior_factor),
                np.sum(self.prior_factor * self.prior_probs),
            )
        )

    def forward(self, data_ab_quant, axis=1):
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if axis == 0:
            return corr_factor[na(), :]
        elif axis == 1:
            return corr_factor[:, na(), :]
        elif axis == 2:
            return corr_factor[:, :, na(), :]
        elif axis == 3:
            return corr_factor[:, :, :, na()]


# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    """Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function"""
    if np.array(inds).size == 1:
        if inds == val:
            return True
    return False


def na():  # shorthand for new axis
    return np.newaxis
