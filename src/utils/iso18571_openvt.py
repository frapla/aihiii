import numpy as np

from scipy.spatial.distance import cdist

from dtwalign import dtw_low
from dtwalign.step_pattern import UserStepPattern
from dtwalign.window import SakoechibaWindow

"""
Python implementation of ISO/TS 18571
https://www.iso.org/standard/62937.html
ISO/TS 18571:2014 provides validation metrics and rating procedures to be used to calculate the level of
correlation between two non-ambiguous signals obtained from a physical test and a computational model, and
is aimed at vehicle safety applications. The objective comparison of time-history signals of model and test
is validated against various loading cases under different types of physical loads such as forces, moments,
and accelerations. However, other applications might be possible too, but are not within the scope of
ISO/TS 18571:2014.

Script from OpenVT 2024-05-23
https://openvt.eu/validation-metrics/ISO18571.git

MIT License
Copyright (c) 2022 Vehicle Safety Institute, Graz University of Technolgy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class CurveLengthError(Exception):
    pass


class ISO18571:
    """This class is used to calculate the Objective Rating Metric for non-ambiguous signals as defined in
        ISO/TS 18571

    Methods of the Class:
        - _cross_correlation:
        - _compute_magnitude:
        - _get_shifted_curve_and_pr:
        - corridor_rating:
        - phase_rating:  calculates the phase rating according to ISO18571
        - magnitude_rating: calculates the magnitude rating according to ISO18571
        - slope_rating: calculates the slope rating according to ISO18571
        - overall_rating: calculates the weighted overall rating according to ISO18571

    Attributes of the Class:
        - method (str): Indicates the method on which the Objective rating method is based on
        - comparison_curve (np.ndarray): xxx
        - reference_curve (np.ndarray): xxx
        - max_shift (float): xxx
        - cae_ts (np.ndarray): xxx
        - t_ts (np.ndarray): xxx
        - n_eps (int):
        - rho_e (float):

    """

    def __init__(
        self,
        reference_curve: np.ndarray,
        comparison_curve: np.ndarray,
        k_z: float = 2.0,
        k_p: int = 1,
        k_m: int = 1,
        eps_m: float = 0.50,
        e_s: float = 2.0,
        init_min: float = 0.8,
        a_0: float = 0.05,
        b_0: float = 0.5,
        w_z: float = 0.4,
        w_p: float = 0.2,
        w_m: float = 0.2,
        w_s: float = 0.2,
    ):
        """Constructor of ISO18571

        Args:
            reference_curve (np.ndarray): numpy array of shape (n, 2). First column must contain the time vector in [s]
                                          with sampling frequency of 10 kHz. Second column contains the reference curve
            comparison_curve (np.ndarray): numpy array of shape (n, 2). First column must contain the time vector in [s]
                                          with sampling frequency of 10 kHz. Second column contains the comparison curve
            k_z (float):
            k_p (int):
            k_m (int):
            eps_m (float):
            e_s (float):
            init_min (float):
            a_0  (float):
            b_0  (float):
            w_z  (float):  Weighting factor of the corridor score
            w_p  (float):  Weighting factor of the phase score
            w_m  (float):  Weighting factor of the magnitude score
            w_s  (float):  Weighting factor of the slope score

        Raises:
            ValueError: if sum of weighting factors (w_z, w_m, w_p, w_s) is unequal to 1
        """

        self.reference_curve = reference_curve
        self.comparison_curve = comparison_curve

        # sanity checks for the two curves
        if self.reference_curve.shape != self.comparison_curve.shape:
            raise CurveLengthError("Curves are not equal in size/dimension.\nInterpolation not implemented. ")

        self._k_z = k_z
        if self._k_z not in [1, 2, 3]:
            raise ValueError("k_z has to be 1, 2, or 3")

        self._k_p = k_p
        if self._k_p not in [1, 2, 3]:
            raise ValueError("k_p has to be 1, 2, or 3")

        self._k_m = k_m
        if self._k_m not in [1, 2, 3]:
            raise ValueError("k_m has to be 1, 2, or 3")

        self._eps_m = eps_m
        self._e_s = e_s

        self._init_min = init_min

        self._a_0 = a_0
        self._b_0 = b_0

        self._w_z = w_z
        self._w_p = w_p
        self._w_m = w_m
        self._w_s = w_s

        weights_sum = self._w_z + self._w_m + self._w_p + self._w_s
        if weights_sum != 1:
            raise ValueError("Sum of weighting factors (w_z, w_m, w_p, w_s) is " + str(weights_sum) + ", but must be 1")

        self._max_shift = round(1.0 - self._init_min, 2)

        self._cae_ts, self._t_ts, self._n_eps, self._rho_e = self._get_shifted_curve_and_pr()

    def _get_shifted_curve_and_pr(self) -> (np.ndarray, np.ndarray, float, float):
        """Calculates the shifted curves, yielded by the highest correlation coefficients

        Returns: The shifted curves cae_ts, t_ts, the maximum cross correlation rho_e and the shifting values,
        which yields rho_e
        """

        rho_l_list, rho_r_list, lag_vec_left, lag_vec_right = ISO18571._cross_correlation(
            self.comparison_curve[:, 1], self.reference_curve[:, 1], self._max_shift
        )
        rho_l_max = max(rho_l_list)
        rho_l_max_idx = np.argmax(rho_l_list)
        rho_r_max = max(rho_r_list)
        rho_r_max_idx = np.argmax(rho_r_list)

        # maximum cross correlation
        rho_e = max(rho_l_max, rho_r_max)

        if rho_r_max > rho_l_max:
            # The number of the time shifting steps, which yields rho_e
            n_eps = lag_vec_right[rho_r_max_idx]

            # shifting to the right
            if n_eps != 0:
                cae_ts = self.comparison_curve[-n_eps:, :]
                t_ts = self.reference_curve[:n_eps, :]
            else:
                cae_ts = self.comparison_curve
                t_ts = self.reference_curve

        else:
            # The number of the time shifting steps, which yields rho_e
            n_eps = lag_vec_left[rho_l_max_idx]

            # shifting to the left
            if n_eps != 0:
                cae_ts = self.comparison_curve[:-n_eps, :]
                t_ts = self.reference_curve[n_eps:, :]
            else:
                cae_ts = self.comparison_curve
                t_ts = self.reference_curve

        return cae_ts, t_ts, n_eps, rho_e

    @staticmethod
    def _cross_correlation(a1: np.array, a2: np.array, max_shift: float) -> (np.array, np.array, np.array, np.array):
        """Returns a list of cross correlations, for the shifted curves a1 and a2, the maximum cross correlation
        E defines the phase error. The cross correlation is calculated by using Pearson product-moment correlation
        coefficients.

        Args:
            a1 (np.array): 1-D array containing observations (a1 and a2 must be of equal length)
            a2 (np.array): 1-D array containing observations (a1 and a2 must be of equal length)
            max_shift (float): Maximum allowable percentage of time shift

        Returns:
            A list of correlation coefficients, for the right resp. left shifted curve.

        Raises:
            ValueError: if max_shift is not in the range of [0, 1].
        """

        window_size = int(np.floor(len(a1) * max_shift) + 1)
        left_idx = np.arange(-window_size, 0)
        right_idx = np.arange(0, window_size)

        rho_l_list = np.zeros(shape=left_idx.shape)
        rho_r_list = np.zeros(shape=left_idx.shape)

        for idx, lag in enumerate(left_idx):
            idx_lower_a1 = max(lag, 0)
            idx_lower_a2 = max(-lag, 0)
            idx_upper_a1 = min(len(a1), len(a1) + lag)
            idx_upper_a2 = min(len(a2), len(a2) - lag)
            b1 = a1[idx_lower_a1:idx_upper_a1]
            b2 = a2[idx_lower_a2:idx_upper_a2]
            c = np.corrcoef(b1, b2)[0][-1]
            rho_l_list[idx] = c

        for idx, lag in enumerate(right_idx):
            idx_lower_a1 = max(lag, 0)
            idx_lower_a2 = max(-lag, 0)
            idx_upper_a1 = min(len(a1), len(a1) + lag)
            idx_upper_a2 = min(len(a2), len(a2) - lag)
            b1 = a1[idx_lower_a1:idx_upper_a1]
            b2 = a2[idx_lower_a2:idx_upper_a2]
            c = np.corrcoef(b1, b2)[0][-1]
            rho_r_list[idx] = c

        return np.array(rho_l_list), np.array(rho_r_list), -left_idx, -right_idx

    @staticmethod
    def _compute_magnitude(x, y, window_size):
        """

        Args:
            x:
            y:
            window_size:

        Returns:

        """

        n = x.shape[0]
        dtw_window_size = int(np.ceil(window_size * n))

        # pattern matches the documentation
        pattern_info = [
            dict(indices=[(-1, 0), (0, 0)], weights=[1]),
            dict(indices=[(0, -1), (0, 0)], weights=[1]),
            dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        ]
        normalize_guide = "none"
        user_step_pattern = UserStepPattern(pattern_info, normalize_guide=normalize_guide)

        X = cdist(x[:, np.newaxis], y[:, np.newaxis], metric="sqeuclidean")

        # argument size passed as dtw_window_size - 1 due to the window size initialization : which is
        # the following in SakoechibaWindow() :  np.abs(xx[:,np.newaxis] - yy[np.newaxis, :]) <= size
        window = SakoechibaWindow(X.shape[0], X.shape[1], size=dtw_window_size - 1)
        res = dtw_low(X, window=window, pattern=user_step_pattern)

        return x[res.path[:, 0]], y[res.path[:, 1]]

    def corridor_rating(self, ndigits: int = 3):
        """Returns the corridor rating for the comparison_curve and the reference_curve

        The corridor score indicates the deviation of the two curves by means of corridor fitting.
        inner and outer corridors form 3 different zones. In the most inner, the fit is rated with 1, in the most
        outer zone, the fit is 0. In between, the value of the fit is affected by the K, which indicates
        the regression relationship. a_0 and b_0 are used to calculate the inner and the outer corridor,
        defining the relative half widths of both.

        Args:
            ndigits (int): precision of ndigits. If negative result is not rounded.

        Returns: the corridor rating, with a precision of ndigits

        """

        t_norm = max(np.abs(self.reference_curve[:, 1]))
        inner_corridor = self._a_0 * t_norm
        outer_corridor = self._b_0 * t_norm

        abs_diff = np.absolute(np.subtract(self.reference_curve, self.comparison_curve))

        # regression for in between corridor
        c_i = np.array(pow(((outer_corridor - abs_diff[:, 1]) / (outer_corridor - inner_corridor)), self._k_z))

        # points in the inner corridor are rated with 1
        c_i[abs_diff[:, 1] < inner_corridor] = 1

        # points in the outer corridor are rated with 0
        c_i[abs_diff[:, 1] > outer_corridor] = 0

        z = sum(c_i) / len(abs_diff)

        if ndigits < 0:
            return z

        return round(z, ndigits=ndigits)

    def phase_rating(self, ndigits: int = 3):
        """Returns the phase score e_p, which is in the range 0 to 1.
         1, if there is no shift between reference and comparison curve
         0, if the maximum allowable time shift threshold has been exceeded.
         In between, the value is calculated by a regression model, depending on k_p

        Args:
            ndigits (int): precision of ndigits. If negative result is not rounded.

        Returns: the phase rating, with a precision of ndigits
        """
        curve_size = self.reference_curve.shape[0]

        # maximum allowable time shift threshold
        # eps_p * N
        max_allowable_time_shift_threshold = curve_size * self._max_shift  # int(round(curve_size * max_shift))

        if self._n_eps == 0:
            e_p = 1
        elif abs(self._n_eps) >= max_allowable_time_shift_threshold:
            e_p = 0
        else:
            e_p = ((max_allowable_time_shift_threshold - abs(self._n_eps)) / max_allowable_time_shift_threshold) ** self._k_p

        if ndigits < 0:
            return e_p

        return round(e_p, ndigits=ndigits)

    def magnitude_rating(self, ndigits: int = 3):
        """Returns the magnitude rating for the comparison_curve and the reference_curve

        Args:
             ndigits (int): precision of ndigits. If negative result is not rounded.

         Returns: the magnitude rating, with a precision of ndigits
        """

        cae_ts_w, t_ts_w = ISO18571._compute_magnitude(self._cae_ts[:, 1], self._t_ts[:, 1], window_size=0.1)

        e_mag = np.linalg.norm(cae_ts_w - t_ts_w, ord=1) / np.linalg.norm(t_ts_w, ord=1)

        if e_mag == 0:
            score_mag = 1
        elif e_mag > self._eps_m:
            score_mag = 0
        else:
            score_mag = ((self._eps_m - e_mag) / self._eps_m) ** self._k_m

        if ndigits < 0:
            return score_mag

        return round(score_mag, ndigits=ndigits)

    def slope_rating(self, ndigits: int = 3):
        """Returns the slope rating for the comparison_curve and the reference_curve

        Args:
             ndigits (int): precision of ndigits. If negative result is not rounded.

         Returns: the slope rating, with a precision of ndigits
        """

        # central difference
        cae_ts_0_d = np.gradient(self._cae_ts[:, 1], 0.0001)
        t_ts_0_d = np.gradient(self._t_ts[:, 1], 0.0001)

        cae_ts_d = np.zeros(len(cae_ts_0_d))
        t_ts_d = np.zeros(len(t_ts_0_d))

        # case 1/1
        for idx, nr in enumerate([1, 3, 5, 7]):
            idx_begin = idx
            idx_end = (-1) * idx - 1
            cae_ts_d[idx_begin] = 1 / nr * np.sum(cae_ts_0_d[0:nr])
            cae_ts_d[idx_end] = 1 / nr * np.sum(cae_ts_0_d[-nr:])
            t_ts_d[idx_begin] = 1 / nr * np.sum(t_ts_0_d[0:nr])
            t_ts_d[idx_end] = 1 / nr * np.sum(t_ts_0_d[-nr:])

        # case 1/9
        nr = 9
        cae_ts_d[4:-4] = np.convolve(cae_ts_0_d, np.ones(nr) / nr, mode="valid")
        t_ts_d[4:-4] = np.convolve(t_ts_0_d, np.ones(nr) / nr, mode="valid")

        e_slope = (np.linalg.norm((cae_ts_d - t_ts_d), ord=1)) / (np.linalg.norm(t_ts_d, ord=1))

        if e_slope <= 0:
            slope_score = 1
        elif e_slope >= self._e_s:
            slope_score = 0
        else:
            slope_score = (self._e_s - e_slope) / self._e_s

        if ndigits < 0:
            return slope_score

        return round(slope_score, ndigits=ndigits)

    def overall_rating(self, ndigits: int = 3) -> float:
        """Returns the overall rating for the comparison_curve and the reference_curve

         Combines the four metric ratings corridor, phase, magnitude and slope to a single number.
         Each rating is weighted with an according weighting factor as indicated in the ISO document.

        Args:
             ndigits (int): precision of ndigits. If negative result is not rounded.

         Returns: overall_rating, which indicates the objective correlation of the analyzed signals, with a
                  precision of ndigits
        """

        z = self.corridor_rating(ndigits=-1)
        e_p = self.phase_rating(ndigits=-1)
        e_m = self.magnitude_rating(ndigits=-1)
        e_s = self.slope_rating(ndigits=-1)

        overall_rating = self._w_z * z + self._w_p * e_p + self._w_m * e_m + self._w_s * e_s

        if ndigits < 0:
            return overall_rating

        return round(overall_rating, ndigits=ndigits)
