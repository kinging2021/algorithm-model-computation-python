import pandas as pd
import matplotlib.pyplot as plt
from algorithm.base.fitting.ee_curve_gsb import call

for i in [5, 6, 7]:
    file = 'GSB01-2019.%d.xlsx' % i
    filename = '/home/junze/jupyter/data/能效分析/' + file
    dh = pd.read_excel(filename, sheet_name='一级指标-1号锅炉单耗')
    fhl = pd.read_excel(filename, sheet_name='一级指标-#1锅炉负荷率')
    dh.rename(columns={'时间': 'time', '采集值': 'dh_cjz', '平均值': 'dh_mean', '最大值': 'dh_max', '最小值': 'dh_min'}, inplace=True)
    fhl.rename(columns={'时间': 'time', '采集值': 'fhl_cjz', '平均值': 'fhl_mean', '最大值': 'fhl_max', '最小值': 'fhl_min'},
               inplace=True)
    data = pd.merge(dh, fhl, on='time')

    dh_list = data['dh_mean'].to_list()
    fhl_list = data['fhl_mean'].to_list()

    param = {
        'y': dh_list,
        'x': fhl_list,
        'x_window': 5,
        'x_range': [0, 100],
        'y_range': [50, 300],
        'degree': 6,
        # 'bounds': [False, False, False, False],
        # 'out_size': 100,
        # 'min_num_sample': 5,
        # 'clamped': True,
    }

    c = call(param=param)

    plt.figure()
    plt.title(file)

    plt.scatter(c.data[:, 0], c.data[:, 1], marker='.', label="data_pts", s=15, color='red')
    plt.scatter(c.sample_points[:, 0], c.sample_points[:, 1], label='ctl_points', marker='x', s=20, color='green')
    plt.plot(c.eval_points[:, 0], c.eval_points[:, 1], label='mid')
    # plt.plot(c.eval_lower_pts['x'], c.eval_lower_pts['y'], label='lower')
    plt.legend()
    plt.show()
