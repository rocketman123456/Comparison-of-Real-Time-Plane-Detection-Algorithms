import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from classes import Result
matplotlib.rcParams.update(
    {'font.size': 26, 'figure.subplot.bottom': 0.06, 'figure.subplot.top': 0.95})
root = 'Stanford3dDataset_v1.2_Aligned_Version'
# algo -> [[size, pre, calc, post]]
data = {'3DKHT': [],
        'RSPD': [],
        'OPS': [],
        'OBRG': []}
st = [0, 0]
k = 0
for i in range(1, 7):
    area = os.path.join(root, f'Area_{i}')
    for scene in os.listdir(area):
        if scene == 'results' or 'nope' in scene or 'DS' in scene:
            continue
        cloudfile = f'{scene}.txt'
        size = os.path.getsize(os.path.join(area, scene, cloudfile))
        # if (size/1000000) > 39:
        #     continue
        with open(os.path.join(area, scene, cloudfile)) as file:
            st[0] += len(file.readlines())
            st[1] += (size/1000000)
        k += 1
        # if size/1000000 > 250:
        #     continue
        for algo in data.keys():
            result = Result.from_file(os.path.join(
                area, scene, 'results', f'{scene}_{algo}.out'))
            if result.algorithm =='OBRG' and result.time_per_plane > 1000:
                continue
            data[algo].append(
                [size, result.time_total, result.time_per_plane, result.time_per_sample, result.precision, result.recall, result.f1])
print(f'{st = }')
print(f'{round(st[0]/k)}/{round(st[1]/k)}')
# exit()
fig = plt.figure(figsize=[80, 30])
max_pre = -1
max_calc = -1
min_pre = 99
min_calc = 99
for da in data.values():
    d = np.array(da.copy())
    if (mp := max(d[:, 1])) > max_pre:
        max_pre = mp
    if (mp := max(d[:, 2])) > max_calc:
        max_calc = mp
    if (mp := min(d[:, 1])) < min_pre:
        min_pre = mp
    if (mp := min(d[:, 2])) < min_calc:
        min_calc = mp

print(f'{max_pre = }')
for i, algo in enumerate(data.keys()):
    d = data[algo].copy()
    d.sort(key=lambda x: x[0])
    d = np.array(d)
    print(f'{algo:10}:{(sum(d[:,0])/len(d))/1000000}')
    print(f'{algo:10}:{sum(d[:,1])/len(d)}')
    print(f'{algo:10}:{sum(d[:,2])/len(d)}')
    print(f'{algo:10}:{sum(d[:,3])/len(d)}')
    print(f'{algo:10}:{sum(d[:,4])/len(d)}')
    print(f'{algo:10}:{sum(d[:,5])/len(d)}')
    print(f'{algo:10}:{sum(d[:,6])/len(d)}')
    d = d * [1/1000000, 1, 1, 1, 1, 1, 1]
    ax = fig.add_subplot(len(data.keys()), 1, i+1)
    if algo == '3DKHT':
        algo = "3D-KHT"
    ax.set_title(algo)
    # ax.set_yscale('log')
    # ax2 = ax.twinx()
    # ax.set_xscale('log')
    # ax.set_ylim(min(min_pre, min_calc), max(max_calc, max_pre)*1.1)
    # ax2.set_ylim(min(min_pre, min_calc), max(max_calc, max_pre)*1.1)
    # ax.scatter(d[:, 0], d[:, 1], s=14)  # label='$t_{pre}$')
    # ax.scatter(d[:, 0], d[:, 2], s=14)  # label='$t_{calc}$')
    ax.plot(d[:, 0], d[:, 1], marker='.', label='$t_{pre}$')
    ax.plot(d[:, 0], d[:, 2], marker='.', label='$t_{calc}$')
    if i > 1:
        ax.plot(d[:, 0], d[:, 3], marker='.', label='$t_{post}$')
        ax.plot(d[:, 0], np.sum(d[:, 1:4], axis=1),
                marker='.',  label='$t_{tot}$')
    else:
        ax.plot(-1, -1, marker='.',  label='$t_{post}$')
        ax.plot(d[:, 0], np.sum(d[:, 1:3], axis=1),
                marker='.',  label='$t_{tot}$')

    # ax.set_yticks([0.0, 0.01,0.1,1,10,100])
    # ax.set_yticklabels([0, 0.1, 1, 10, 100, 200])
    # ax2.set_yticklabels([0, 0.1, 1, 10, 100, 200])
    # ticks =[float(format(x, 'f')) for x in ax.get_yticks()]
    # print(ticks)
    # ax.set_yticks(ticks)
    # ax.set_yticks([0, 10, 100, 200,500, 1000])
    # ax.set_yticklabels(["0","1","10","100",f'{max(max_calc,max_pre)}'])
    ax.set_yscale('log')
    ax.set_yticks([min(min_pre, min_calc), 1, 10,100,700]) #round(max(max_calc, max_pre)*1.1)])
    print(ax.get_yticks())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(min(min_pre, min_calc),700)
    ax.grid(axis='y')

    # ax.ticklabel_format(useOffset=False)
    if i != 3:
        ax.get_xaxis().set_visible(False)
    # ax.set_xscale('log')
fig.text(0.06, 0.5, '$t_{pre}(s), t_{calc}(s), t_{post}(s), t_{tot}(s)$',
         ha='center', va='center', rotation='vertical')
fig.text(0.5, 0, 'File Size(mb)', ha='center',
         va='bottom', rotation='horizontal')

fig.axes[0].legend(loc='lower right', fancybox=True, shadow=True, ncol=5)

plt.show()