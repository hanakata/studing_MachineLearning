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
