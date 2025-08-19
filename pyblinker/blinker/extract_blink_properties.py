import numpy as np
import pandas as pd


class BlinkProperties:
    """
    Return a structure with blink shapes and properties for individual blinks
    % Return a structure with blink shapes and properties for individual blinks
    %
    % Parameters:
    %     signalData    signalData structure
    %     params        params structure with parameters
    %     blinkProps    (output) structure with the blink properties
    %     blinkFits     (output) structure with the blink landmarks
    """

    def __init__(self, candidate_signal, df, srate, params, *, fitted: bool = True):
        """Initializes BlinkProperties object to calculate blink features.

        This class calculates various properties of detected blinks based on
        the input candidate_signal, DataFrame of blink fits, sampling rate, and parameters.
        It initializes blink velocity, durations, amplitude-velocity ratios,
        and time-related features.

        Parameters
        ----------
        candidate_signal : numpy.ndarray
            The raw signal candidate_signal from which blinks were detected.
        df : pandas.DataFrame
            DataFrame containing blink fit parameters, expected to have columns like
            'left_base', 'right_base', 'left_zero', 'right_zero', 'right_x_intercept', 'left_x_intercept',
            'left_base_half_height', 'right_base_half_height', 'left_zero_half_height', 'right_zero_half_height',
            'max_blink', 'max_value', 'aver_right_velocity', 'aver_left_velocity', 'x_intersect', 'y_intersect',
            'left_x_intercept_int', 'right_x_intercept_int', 'start_shut_tst', 'peaks_pos_vel_base', 'peaks_pos_vel_zero'.
        srate : float
            Sampling rate of the signal candidate_signal in Hz.
        params : dict
            Dictionary of parameters, expected to contain keys:
                - 'shut_amp_fraction': Fraction of maximum amplitude for shut time calculation.
                - 'z_thresholds': Z-score thresholds (structure of thresholds is assumed to be handled internally by methods using it).
        fitted : bool, optional
            If ``True`` additional features requiring blink fitting are computed.
            Defaults to ``True``.
        """
        self.signal_len = None
        self.blink_velocity = None
        self.candidate_signal = candidate_signal
        self.df = df
        self.srate = srate
        self.shut_amp_fraction = params["shut_amp_fraction"]
        self.p_avr_threshold = params["p_avr_threshold"]
        self.z_thresholds = params["z_thresholds"]

        self.fitted = fitted

        self.df_res = []
        self.reset_index()
        self.set_blink_velocity()
        self.set_blink_duration()

        self.set_blink_amp_velocity_ratio_zero_to_max()
        self.amplitude_velocity_ratio_base()
        if fitted:
            self.amplitude_velocity_ratio_tent()
        self.time_zero_shut()
        self.time_base_shut()
        self.extract_other_times()

    def reset_index(self):
        self.df.reset_index(drop=True, inplace=True)

    def set_blink_velocity(self):
        self.signal_len = self.candidate_signal.shape[0]
        self.blink_velocity = np.diff(self.candidate_signal)

    def set_blink_duration(self):
        """Add blink durations in seconds to ``self.df`` using snake_case columns."""

        constant = 1  # Constant for matching Matlab output

        self.df["duration_base"] = (
            self.df["right_base"] - self.df["left_base"]
        ) / self.srate
        self.df["duration_zero"] = (
            self.df["right_zero"] - self.df["left_zero"]
        ) / self.srate

        if self.fitted:
            self.df["duration_tent"] = (
                self.df["right_x_intercept"] - self.df["left_x_intercept"]
            ) / self.srate
            self.df["duration_half_base"] = (
                (self.df["right_base_half_height"] - self.df["left_base_half_height"]) + constant
            ) / self.srate
            self.df["duration_half_zero"] = (
                (self.df["right_zero_half_height"] - self.df["left_zero_half_height"]) + constant
            ) / self.srate

    def compute_amplitude_velocity_ratio(
        self, start_key, end_key, ratio_key, aggregator="max", idx_col=None
    ):
        """
        Internal helper that computes amplitude-velocity ratio between start_key and end_key.
        aggregator can be 'max' or 'min' to pick the velocity extremes.
        idx_col can store the index of extreme velocity if desired.
        """
        start_vals = self.df[start_key].to_numpy().astype(int)
        end_vals = self.df[end_key].to_numpy().astype(int)
        blink_vel = self.blink_velocity

        lengths = (end_vals - start_vals + 1).astype(int)
        max_len = lengths.max()
        offsets = np.arange(max_len)[None, :]
        mask = offsets < lengths[:, None]

        all_indices = start_vals[:, None] + offsets
        all_indices = all_indices[mask].astype(int)

        row_idx_all = np.repeat(np.arange(len(lengths)), lengths)
        velocities = blink_vel[all_indices]

        temp_df = pd.DataFrame(
            {"row_idx": row_idx_all, "velocity": velocities, "index": all_indices}
        )
        if aggregator == "max":
            idx_extreme = temp_df.groupby("row_idx")["velocity"].idxmax()
        else:
            idx_extreme = temp_df.groupby("row_idx")["velocity"].idxmin()

        df_extreme = temp_df.loc[idx_extreme].sort_values("row_idx")
        ratio_vals = (
            100
            * abs(
                self.candidate_signal[self.df["max_blink"].to_numpy()]
                / df_extreme["velocity"].to_numpy()
            )
            / self.srate
        )

        self.df[ratio_key] = ratio_vals
        if idx_col:
            self.df[idx_col] = df_extreme["index"].to_numpy()

    def compute_neg_amp_vel_ratio_zero(self):
        """
        Computes and sets negative amplitude-velocity ratio from ``max_blink`` to ``right_zero`` in DataFrame.
        Computes and sets positive amplitude-velocity ratio from ``left_zero`` to ``max_blink`` in DataFrame.

        """
        self.compute_amplitude_velocity_ratio(
            start_key="max_blink",
            end_key="right_zero",
            ratio_key="neg_amp_vel_ratio_zero",
            aggregator="min",
        )

    def compute_pos_amp_vel_ratio_zero(self):
        """
        Computes and sets positive amplitude-velocity ratio from ``left_zero`` to ``max_blink`` in DataFrame.
        """

        self.compute_amplitude_velocity_ratio(
            start_key="left_zero",
            end_key="max_blink",
            ratio_key="pos_amp_vel_ratio_zero",
            aggregator="max",
            idx_col="peaks_pos_vel_zero",
        )

    def set_blink_amp_velocity_ratio_zero_to_max(self):
        """ "Computes and sets both positive and negative amplitude-velocity ratios (zero-to-max)."""
        self.compute_pos_amp_vel_ratio_zero()
        self.compute_neg_amp_vel_ratio_zero()

    def compute_pos_amp_vel_ratio_base(self):
        """Computes and sets positive amplitude-velocity ratio from ``left_base`` to ``max_blink`` in DataFrame."""
        self.compute_amplitude_velocity_ratio(
            start_key="left_base",
            end_key="max_blink",
            ratio_key="pos_amp_vel_ratio_base",
            aggregator="max",
            idx_col="peaks_pos_vel_base",
        )

    def compute_neg_amp_vel_ratio_base(self):
        """Computes and sets negative amplitude-velocity ratio from ``max_blink`` to ``right_base`` in DataFrame."""
        self.compute_amplitude_velocity_ratio(
            start_key="max_blink",
            end_key="right_base",
            ratio_key="neg_amp_vel_ratio_base",
            aggregator="min",
        )

    def amplitude_velocity_ratio_base(self):
        """
        Blink amplitude-velocity ratio from base to max
        :return:
        """
        self.compute_pos_amp_vel_ratio_base()
        self.compute_neg_amp_vel_ratio_base()

    def amplitude_velocity_ratio_tent(self):
        """
         Blink amplitude-velocity ratio estimated from tent slope
        :return:
        """
        self.df["neg_amp_vel_ratio_tent"] = (
            100
            * abs(
                self.candidate_signal[self.df["max_blink"]]
                / self.df["aver_right_velocity"]
            )
            / self.srate
        )
        self.df["pos_amp_vel_ratio_tent"] = (
            100
            * abs(
                self.candidate_signal[self.df["max_blink"]]
                / self.df["aver_left_velocity"]
            )
            / self.srate
        )

    @staticmethod
    def compute_time_shut(
        row, data, srate, shut_amp_fraction, key_prefix, default_no_thresh
    ):
        """
        Compute shut duration using specified landmarks for a blink.
        Basiclly, we combine what before having seperate function but mostly overlap syntax

        This code will be use to calculate the  timeShutBase and timeShutZero
        Parameters:
          row : pandas.Series
              A row containing the blink candidate_signal.
          data : array-like
              Signal candidate_signal from which to compute the duration.
          srate : float
              Sampling rate.
          shut_amp_fraction : float
              Fraction of the max amplitude used to determine the threshold.
          key_prefix : str
              The suffix used in the column names to determine the left/right boundaries.
              Use 'Zero' for zero-crossing landmarks or 'Base' for base landmarks.
          default_no_thresh : scalar
              The value to return if the signal never meets the threshold.
              For example, np.nan for zero-crossing or 0 for base landmarks.

        Returns:
          float
              The computed shut duration in seconds.
        """
        col_left = f"left_{key_prefix.lower()}"
        col_right = f"right_{key_prefix.lower()}"
        left = int(row[col_left])
        right = int(row[col_right])
        threshold = shut_amp_fraction * row["max_value"]
        data_slice = data[left : right + 1]

        # Find the start index where the signal first reaches/exceeds the threshold.
        cond = data_slice >= threshold
        if not cond.any():
            return default_no_thresh
        start_idx = cond.argmax()

        # Find the first index after start_idx where the signal drops below the threshold.
        cond = data_slice[start_idx + 1 :] < threshold
        end_idx = cond.argmax() + 1 if cond.any() else np.nan
        return end_idx / srate

    def time_zero_shut(self):
        """
        Time zero shut
        :return:
        """
        self.df["closing_time_zero"] = (
            self.df["max_blink"] - self.df["left_zero"]
        ) / self.srate
        self.df["reopening_time_zero"] = (
            self.df["right_zero"] - self.df["max_blink"]
        ) / self.srate

        self.df["time_shut_base"] = self.df.apply(
            lambda row: self.compute_time_shut(
                row,
                self.candidate_signal,
                self.srate,
                self.shut_amp_fraction,
                key_prefix="Base",
                default_no_thresh=0,
            ),
            axis=1,
        )

    @staticmethod
    def compute_time_shut_tent(row, candidate_signal, srate, shut_amp_fraction):

        left = int(row["left_x_intercept"])
        right = int(row["right_x_intercept"]) + 1
        max_val = row["max_value"]
        amp_threshold = shut_amp_fraction * max_val
        data_slice = candidate_signal[left:right]

        cond_start = data_slice >= amp_threshold
        if not cond_start.any():
            return 0

        start_idx = np.argmax(cond_start)
        cond_end = data_slice[start_idx:-1] < amp_threshold
        end_shut = np.argmax(cond_end) if cond_end.any() else np.nan
        return end_shut / srate

    def time_base_shut(self):
        """
        Time base shut
        :return:
        """

        self.df["time_shut_base"] = self.df.apply(
            lambda row: self.compute_time_shut(
                row,
                self.candidate_signal,
                self.srate,
                self.shut_amp_fraction,
                key_prefix="Base",
                default_no_thresh=0,
            ),
            axis=1,
        )

        if self.fitted:
            self.df["closing_time_tent"] = (
                self.df["x_intersect"] - self.df["left_x_intercept"]
            ) / self.srate
            self.df["reopening_time_tent"] = (
                self.df["right_x_intercept"] - self.df["x_intersect"]
            ) / self.srate

            self.df["time_shut_tent"] = self.df.apply(
                lambda row: self.compute_time_shut_tent(
                    row,
                    self.candidate_signal,
                    self.srate,
                    self.shut_amp_fraction,
                ),
                axis=1,
            )

    def get_argmax_val(self, row):

        left = row["left_x_intercept_int"]
        right = row["right_x_intercept_int"] + 1
        start = row["start_shut_tst"]
        max_val = row["max_value"]
        subset = self.candidate_signal[left:right][start:-1]
        try:
            return np.argmax(subset < self.shut_amp_fraction * max_val)
        except ValueError:
            return np.nan

    def extract_other_times(self):

        self.df["peak_max_blink"] = self.df["max_value"]
        if self.fitted:
            self.df["peak_max_tent"] = self.df["y_intersect"]
            self.df["peak_time_tent"] = self.df["x_intersect"] / self.srate
        self.df["peak_time_blink"] = self.df["max_blink"] / self.srate

        peaks_with_len = np.append(
            self.df["max_blink"].to_numpy(), len(self.candidate_signal)
        )
        self.df["inter_blink_max_amp"] = np.diff(peaks_with_len) / self.srate

        self.df["inter_blink_max_vel_base"] = (
            self.df["peaks_pos_vel_base"] * -1
        ) / self.srate
        self.df["inter_blink_max_vel_zero"] = (
            self.df["peaks_pos_vel_zero"] * -1
        ) / self.srate
