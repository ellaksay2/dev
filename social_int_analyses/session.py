import dill
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.linear_model import LinearRegression
import TwoPUtils

from suite2p.extraction import dcnv
from scipy.interpolate import interp1d as spline
    
class SocIntSession(TwoPUtils.sess.Session):

    def __init__(self, prev_sess=None, **kwargs):

        """

        :param self:
        :param kwargs:
        :return:
        """
        self.trial_info = None
        self.z2t_spline = None
        self.t2z_spline = None
        self.t2x_spline = None
        self.rzone_early = None
        self.rzone_late = None
        # self.novel_arm = None
        self.place_cell_info = {}

        if prev_sess is not None:
            for attr in dir(prev_sess):
                if not attr.startswith('__') and not callable(getattr(prev_sess, attr)):
                    kwargs[attr] = getattr(prev_sess, attr)
                    # setattr(self, attr, getattr(prev_sess, attr))

        super(SocIntSession, self).__init__(**kwargs)




    @classmethod
    def from_file(cls, filename, **kwargs):
        '''
        initialize class from previous instance

        :param filename:
        :return:
        '''
        with open(filename, 'rb') as file:
            return cls(prev_sess=dill.load(file), **kwargs)

    @staticmethod
    def _get_t(t, p0, p1, alpha=.5):
        '''

        :param t:
        :param p0:
        :param p1:
        :param alpha:
        :return:
        '''
        a = (p0 - p1) ** 2
        b = a.sum() ** (alpha * .5)
        return b + t

    @staticmethod
    def _catmulrom(_t, tvec, control_points):
        '''

        :param _t:
        :param tvec:
        :param control_points:
        :return:
        '''
        if tvec[1] <= _t < tvec[2]:
            ind = 0
        elif tvec[2] <= _t < tvec[3]:
            ind = 1
        elif tvec[3] <= _t < tvec[4]:
            ind = 2
        elif tvec[4] <= _t < tvec[5]:
            ind = 3
        elif tvec[5] <= _t < tvec[6]:
            ind = 4
        else:
            _t = tvec[2]
            ind = 1
        #     print(ind)
        p0 = control_points[ind, :]
        p1 = control_points[ind + 1, :]
        p2 = control_points[ind + 2, :]
        p3 = control_points[ind + 3, :]

        t0, t1, t2, t3 = tvec[ind], tvec[ind + 1], tvec[ind + 2], tvec[ind + 3]

        a1 = (t1 - _t) / (t1 - t0) * p0 + (_t - t0) / (t1 - t0) * p1
        a2 = (t2 - _t) / (t2 - t1) * p1 + (_t - t1) / (t2 - t1) * p2
        a3 = (t3 - _t) / (t3 - t2) * p2 + (_t - t2) / (t3 - t2) * p3

        b1 = (t2 - _t) / (t2 - t0) * a1 + (_t - t0) / (t2 - t0) * a2
        b2 = (t3 - _t) / (t3 - t1) * a2 + (_t - t1) / (t3 - t1) * a3

        c = (t2 - _t) / (t2 - t1) * b1 + (_t - t1) / (t2 - t1) * b2
        return c

    def add_pos_binned_trial_matrix(self, ts_name, pos_key='t', min_pos=13, max_pos=43, bin_size=1, mat_only=True, 
                                    **trial_matrix_kwargs):
        """

        :param ts_name:
        :param pos_key:
        :param min_pos:
        :param max_pos:
        :param bin_size:
        :param mat_only:
        :param trial_matrix_kwargs:
        :return:
        """
        super(YMazeSession, self).add_pos_binned_trial_matrix(ts_name, pos_key,
                                                              min_pos=min_pos,
                                                              max_pos=max_pos,
                                                              bin_size=bin_size,
                                                              mat_only=mat_only,
                                                              **trial_matrix_kwargs)

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2
