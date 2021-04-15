import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

fig = plt.figure(figsize=(12, 6))
fig.patch.set_alpha(0.)

plt.rcParams["font.family"] = "Times New Roman"      #全体のフォントを設定
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線を内向きへ
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線を内向きへ
plt.rcParams["xtick.minor.visible"] = True           #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True           #y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1              #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1              #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 0.5              #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 0.5              #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 7                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 7                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 3                 #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 3                 #y軸補助目盛り線の長さ
plt.rcParams["font.size"] = 24                       #フォントの大きさ
plt.rcParams["axes.linewidth"] = 1                 #囲みの太さ
ax = plt.gca()
ax.spines["right"].set_color("#2E2E2E") 
ax.spines["top"].set_color("#2E2E2E")  
ax.spines["left"].set_color("#2E2E2E")    
ax.spines["bottom"].set_color("#2E2E2E") 
ax.patch.set_alpha(0.)
ax.xaxis.label.set_color('#2E2E2E')
ax.yaxis.label.set_color('#2E2E2E')
ax.tick_params(axis = 'x', colors ='#2E2E2E')
ax.tick_params(axis = 'y', colors = '#2E2E2E')
ax.locator_params(axis='x',nbins=5)
ax.locator_params(axis='y',nbins=4)

def convertVoltonT():
    max_vol = 5
    ANA_OUT_RANGE = 8.1 # 0.33X = 0.9, 1X = 2.7, 3X = 8.1
    return 1/ANA_OUT_RANGE
def convertAngle():
    STEP = 2097151.995
    return 2*np.pi/STEP

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gaussfit(x_list,y_list,para_ini):
    para_opt, cov = scipy.optimize.curve_fit(
    gauss, x_list, y_list, para_ini)  # フィッティングパラメータの最適化
    print(para_opt)
    plot_num = 1000
    x_arr = np.arange(min(x_list),max(x_list),(max(x_list)-min(x_list))/plot_num)
    y_arr = gauss(x_arr,*para_opt)
    return x_arr,y_arr

def limitRange(x_list,y_list,x_min,x_max):
    limit_x_list = []
    limit_y_list = []
    for i, x in enumerate(x_list):
        if x > x_min and x < x_max:
            limit_x_list.append(x)
            limit_y_list.append(y_list[i])
    return limit_x_list,limit_y_list

ax.set_xlabel('angle (rad)')
ax.set_ylabel('Bz (nT)')
data = np.loadtxt("data/1.txt", delimiter=",")
x_arr = data[:,0]*convertAngle()%(2*np.pi)
y_arr = data[:,1]*convertVoltonT() 

limit_x_arr, limit_y_arr = limitRange(x_arr,y_arr,5,5.8)
fit_x_arr,fit_y_arr = gaussfit(limit_x_arr, limit_y_arr,[1,5,0.1])
ax.plot(fit_x_arr,fit_y_arr)

limit_x_arr2, limit_y_arr2 = limitRange(x_arr,y_arr,5.9,6.28)
fit_x_arr2,fit_y_arr2 = gaussfit(limit_x_arr2, limit_y_arr2,[1,6,3.1])
ax.plot(fit_x_arr2,fit_y_arr2)

ax.plot(x_arr,y_arr, 'o', markersize='3', color='gray')



plt.show()
