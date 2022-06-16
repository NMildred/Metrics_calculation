from pathlib import Path

DATA_PATH = Path("./data")

MODEL_PATH = Path("./model")

SENSORS = ["s1", "s2", "s3", "x", "y", "z"]

SENSOR_NAME = ['s1_', 's2_', 's3_', 'x_', 'y_', 'z_', 'mag_', "mag1_"]

SENSORS1 = ["s1", "s2", "s3", "x", "y", "z", "mag", "mag1"]

ACT_LIST = ['sit','jog','walk','noact']

FINAL_FEAT = ['s2_peak_len_50_110', 's2_variation', 's2_mean', 'average', 'mag_entropy', 'x_mean', 'x_peak_prom', 
          'y_variation', 'mag_median', 'y_kurtosis', 'x_skew', 'z_skew', 'stdovermean1', 'y_peak_width', 
          'z_kurtosis', 'y_std_div_z_std', 'y_tvar_div_z_tvar', 'y_skew', 's2_median', 'mag_skew', 'y_peak_prom',
          'z_median', 'x_kurtosis', 'x_peak_width', 'y_peak_len_50_110', 'x_std_div_z_std', 'z_peak_prom', 
          'x_variation', 'mag_mean', 's2_skew', 'y_kstatvar']

M3 = 7.336898431604652 #calibration coefficient