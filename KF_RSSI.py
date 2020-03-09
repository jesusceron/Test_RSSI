import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter

csv_df = pd.read_csv('test_RSSI\Libro3.csv')
fig_titles = csv_df.keys()

Q_std = 0.3


def tracker1(x_initial, R_std):
    tracker = KalmanFilter(dim_x=1, dim_z=1)

    tracker.F = np.array([1])
    tracker.u = 0.
    tracker.H = np.array([1])

    tracker.R = R_std
    tracker.Q = Q_std
    tracker.x = np.array([[x_initial]]).T
    tracker.P = 0
    return tracker


i = 0
fig, axs = plt.subplots(len(csv_df.columns), figsize=(20, 30))

for column in csv_df:
    data = csv_df[column]
    x_initial = data[0]
    R_std = np.var(data)

    # run filter
    robot_tracker = tracker1(x_initial, R_std)
    mu, cov, _, _ = robot_tracker.batch_filter(data)

    axs[i].set_title('Beacon: {}'.format(fig_titles[i]))
    axs[i].plot(mu[:, 0])
    axs[i].plot(data)
    axs[i].grid()
    i += 1

plt.show()
