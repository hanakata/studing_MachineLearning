import pandas as pd
# Date列にDateTime型を適用する
df = pd.read_csv('HostLogons-demo.csv', parse_dates=["Date"], infer_datetime_format=True)
print(df)
# Date列とComputerName列が同一の行のTotalLogons列をすべて加算して新たな行列にコピー
df_LogonSum = df.groupby(['Date','ComputerName'])['TotalLogons'].sum().reset_index()

# Date列とTotalLogons列のみを選択する
df_LogonSum = df_LogonSum[['Date','TotalLogons']]
print(df_LogonSum)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 1, 1)
# オリジナルの時系列データの描画
ax.plot(df_LogonSum['Date'],df_LogonSum['TotalLogons'], label="original")

# X軸ラベルの調整
daysFmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(daysFmt)
fig.autofmt_xdate()

plt.grid(True)

plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 1, 1)
# 原系列データの描画
ax.plot(
    df_LogonSum['Date'],
    df_LogonSum['TotalLogons'], 
    label="original"
    )
# 7区間移動平均
ax.plot(
    df_LogonSum['Date'],
    df_LogonSum['TotalLogons'].rolling(7).mean(), 
    label="rolling", 
    ls="dashed"
    )
plt.title('Daily TotalLogons')

# X軸ラベルの調整
daysFmt = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(daysFmt)
fig.autofmt_xdate()

plt.grid(True)

plt.show()

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# モデルにmultiplicativeを指定し、乗算モデルを使用する
result = seasonal_decompose(
    df_LogonSum['TotalLogons'], 
    model='multiplicative', 
    freq=7
    )

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 7), sharex=True)
plt.subplots_adjust(hspace=0.5)

# 原系列
axes[0].set_title('Observed')
axes[0].plot(result.observed)

# 傾向変動
axes[1].set_title('Trend')
axes[1].plot(result.trend)

# 季節変動
axes[2].set_title('Seasonal')
axes[2].plot(result.seasonal)

# 不規則変動
axes[3].set_title('Residual')
axes[3].plot(result.resid)

# グラフの表示
plt.show()

trend = result.trend
trend = pd.DataFrame({'trend': trend, 'date':df_LogonSum.Date})
trend['date'] = pd.to_datetime(trend['date'], format='%Y-%m-%d')
trend = trend.set_index(['date'])
trend = trend.plot()

