import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.core.display import display
import pandas as pd
import scipy.signal as signal
import numpy as np
import torch

# 顔の線を描画するラインの順番を生成する
def getLines():
    # 輪郭
    lines = [[i, i+1] for i in range(16)]
    
    # 口(外)
    lines += [[i, i+1] for i in range(48, 59)]
    lines += [[59, 48]]
    
    # 口(外)
    lines += [[i, i+1] for i in range(60, 67)]
    lines += [[67, 60]]
    
    # 鼻筋
    lines += [[i, i+1] for i in range(27, 30)]
    
    # 鼻
    lines += [[i, i+1] for i in range(31, 35)]
    
    # 眉(左)
    lines += [[i, i+1] for i in range(17, 21)]
    
    # 眉(右)
    lines += [[i, i+1] for i in range(22, 26)]
    
    # 目(左)
    lines += [[i, i+1] for i in range(36, 41)]
    lines += [[41, 36]]
    
    # 目(右)
    lines += [[i, i+1] for i in range(42, 47)]
    lines += [[47, 42]]
    return lines

# 顔の単画像を描画
"""
input
xy: (1, 136)
"""
def plot_face(xy, ax=None, color='b'):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x, y = xy[:68], xy[68:]
    ax.set_aspect('equal')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.max(), y.min())
    
    lines = getLines()
    
    for line in lines:
        ax.plot(x[line], y[line], color="b")
    ax.scatter(x, y, s=3)

# 顔の動画を描画
"""
input
XY: (fps, 136)

output
HTML object
"""
def plot_anim_face(XY, fps=8, save=None):
    X, Y =  XY[:, :68], XY[:, 68:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    def update(i):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.max(), Y.min())
        x, y = X[i], Y[i]
        lines = getLines()
        for line in lines:
            ax.plot(x[line], y[line], color="b")
        ax.scatter(x, y, s=3)
    ani = animation.FuncAnimation(fig, update, len(X),interval=1000/fps)
    if save:
        ani.save(save, writer='imagemagick')
    display(HTML(ani.to_jshtml()))
    plt.close()


# 顔の線を描画するラインの順番を生成する
def getMouthLines():
    lines = []
    # 口(外)
    lines += [[i, i+1] for i in range(11)]
    lines += [[11, 0]]
    
    # 口(外)
    lines += [[i, i+1] for i in range(12, 19)]
    lines += [[19, 12]]
    return lines

def plot_mouth(xy, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x, y = xy[:20], xy[20:]
    ax.set_aspect('equal')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.max(), y.min())
    
    lines = getMouthLines()
    
    for line in lines:
        ax.plot(x[line], y[line], color="b")
    ax.scatter(x, y, s=3)

def plot_anim_mouth(XY, fps=8):
    X, Y =  XY[:, :20], XY[:, 20:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def update(i):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.max(), Y.min())
        x, y = X[i], Y[i]
        lines = getMouthLines()
        for line in lines:
            ax.plot(x[line], y[line], color="b")
        ax.scatter(x, y, s=3)
    ani = animation.FuncAnimation(fig, update, len(X),interval=1000/fps)
    display(HTML(ani.to_jshtml()))
    plt.close()

def getContourLines():
    # 輪郭
    lines = [[i, i+1] for i in range(16)]
    
    # 鼻筋
    lines += [[i, i+1] for i in range(27, 30)]
    
    # 鼻
    lines += [[i, i+1] for i in range(31, 35)]
    
    # 眉(左)
    lines += [[i, i+1] for i in range(17, 21)]
    
    # 眉(右)
    lines += [[i, i+1] for i in range(22, 26)]
    
    # 目(左)
    lines += [[i, i+1] for i in range(36, 41)]
    lines += [[41, 36]]
    
    # 目(右)
    lines += [[i, i+1] for i in range(42, 47)]
    lines += [[47, 42]]
    return lines

def plot_contour(xy, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x, y = xy[:48], xy[48:]
    ax.set_aspect('equal')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.max(), y.min())
    
    lines = getContourLines()
    
    for line in lines:
        ax.plot(x[line], y[line], color="b")
    ax.scatter(x, y, s=3)

def plot_anim_contour(XY, fps=8):
    X, Y =  XY[:, :48], XY[:, 48:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def update(i):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.max(), Y.min())
        x, y = X[i], Y[i]
        lines = getContourLines()
        for line in lines:
            ax.plot(x[line], y[line], color="b")
        ax.scatter(x, y, s=3)
    ani = animation.FuncAnimation(fig, update, len(X),interval=1000/fps)
    display(HTML(ani.to_jshtml()))
    plt.close()
    
# ローパスフィルタを実現
def lowpass(series, fc=8, fs=30, numtaps=256):
    fir_filter = signal.firwin(numtaps=numtaps, cutoff=fc, fs=fs)
    return signal.lfilter(fir_filter, 1, series)

# 前処理関数
"""
input
file_path: OpenFaceのcsv出力ファイルのパス, 
fps: ローパスを掛ける周波数＋前処理後のfps

output
data: (*, 136)
"""
def preprocess(file_path, fps=8):
    df = pd.read_csv(file_path)
    df.loc[df[" confidence"] < 0.7, " gaze_0_x": " AU45_c"] = None
    df_land = df.loc[:, " x_0":" y_67"]
    df_land = df_land.interpolate('values')
    df_land_lowpass = df_land.apply(lambda x: lowpass(x, fc=fps, fs=30), axis=0)
    df_land_lowpass.index = pd.to_timedelta(df[" timestamp"], unit="S")
    df_land_lowpass = df_land_lowpass.dropna()
    df_land_lowpass = df_land_lowpass.resample(f"{1000//fps}ms").mean().reset_index()
    df_land_lowpass.drop(" timestamp", axis=1, inplace=True)
    df_land_lowpass = df_land_lowpass[fps * 5:]

    X, Y = df_land_lowpass.loc[:, " x_0":" x_67"], df_land_lowpass.loc[:, " y_0":" y_67"]
    X, Y = X.apply(lambda x: x - x.mean(), axis=1), Y.apply(lambda x: x - x.mean(), axis=1)
    Y_row_width = Y.max(axis=1) - Y.min(axis=1)
    X, Y = X.apply(lambda x: x/Y_row_width, axis=0), Y.apply(lambda x: x/Y_row_width, axis=0)
    
    data = np.empty((0, 136))
    for i in range(len(X)):
        img = np.hstack([X.iloc[i], Y.iloc[i]]).reshape(1, 136)
        data = np.append(data, img, axis=0)
    return data

# 顔画像データを動画の配列に変換
"""
input
data: preprocessで生成されたデータ
frames: 何フレームを一つにするか

output:
split_dat: (*, frames, 136)
"""
def face_simple_to_movie(data, frames):
    split_data = np.vsplit(data[:-(len(data)%frames)], len(data)//frames)
    split_data = np.array(split_data, dtype=np.float32)
    return split_data

# numpy配列をpytorchのデータローダーに変換
"""
input
data: numpy配列
barch_size: バッチサイズ
shuffle: データローダーをシャッフルするか

output
dataloader: torch.utils.dataloader
"""
def numpy_to_dataloader(data, batch_size=32, shuffle=True):
    dataset = torch.from_numpy(data.astype(np.float32)).clone()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 顔画像データを口と輪郭に分離
def separate_mouth_contour(face_data):
    mouth_data = torch.cat((face_data[:, 48:68], face_data[:, 48+68:68+68]), dim=1)
    contour_data = torch.cat((face_data[:, :48], face_data[:, 68:48+68], face_data[:, 68+68:]), dim=1)
    return mouth_data, contour_data

# 口と輪郭データを顔画像データに結合
def combine_mouth_contour(mouth_data, contour_data):
    mouth_data_x, mouth_data_y = mouth_data[:, :20], mouth_data[:, 20:]
    contour_data_x, contour_data_y = contour_data[:, :48], contour_data[:, 48:]
    face_data = torch.cat((contour_data_x, mouth_data_x, contour_data_y, mouth_data_y), dim=1)
    return face_data