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
        self.novel_arm = None
        self.place_cell_info = {}

        if prev_sess is not None:
            for attr in dir(prev_sess):
                if not attr.startswith('__') and not callable(getattr(prev_sess, attr)):
                    kwargs[attr] = getattr(prev_sess, attr)
                    # setattr(self, attr, getattr(prev_sess, attr))

        super(YMazeSession, self).__init__(**kwargs)

        self._get_pos2t_spline()
        if self.novel_arm is not None:
            if self.novel_arm == -1:
                self.rzone_nov = self.rzone_early
                self.rzone_fam = self.rzone_late
            elif self.novel_arm == 1:
                self.rzone_fam = self.rzone_early
                self.rzone_nov = self.rzone_late

        if isinstance(self.iscell, pd.DataFrame):
            self.mcherry_curated = True
        else:
            self.mcherry_curated = False


