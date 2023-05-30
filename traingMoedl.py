import sys

sys.path.append('/home/aistudio/xiazai')
import re
import paddle
from paddlets.datasets.tsdataset import TSDataset
from paddlets.models.forecasting import LSTNetRegressor
import warnings
import os
warnings.filterwarnings('ignore')
import pandas as pd

station = ["A", "B", "C", "D", "E", "F", "G"]
stastion_Dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
area_Dict = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 2, "F": 2, "G": 3}
area_list = []
workday_holiday_list = []
#

def extract_station(x): # 设置站点
    station = re.compile("([A-Z])站").findall(x)
    if len(station) > 0:
        area_list.append(area_Dict[station[0]])
        return stastion_Dict[station[0]]


def get_MorningAndEvening_peak(x):
    hour = x[0:x.find(":")]
    if int(hour) < 6:
        return 0.0
    elif int(hour) < 7:
        return 1.5
    elif int(hour) < 8:
        return 3
    elif int(hour) < 9:
        return 9
    elif int(hour) < 12:
        return 4
    elif int(hour) < 17:
        return 6
    elif int(hour) < 19:
        return 10
    elif int(hour) < 22:
        return 4
    else:
        return 1.5


df = pd.read_csv("data/客流量数据.csv")
df.interpolate(method="linear", axis=0, inplace=True)
df["时间区段"] = df["时间区段"].apply(lambda x: x.split("-")[0])
df["站点"] = df["站点"].apply(extract_station)
df["时间"] = df["时间"].apply(lambda x: str(x)).str.cat(df['时间区段'], sep=" ")
df["时间"] = pd.to_datetime(df["时间"])
df["MorningAndEveningPeak"] = df["时间区段"].apply(get_MorningAndEvening_peak)
df['dayOfWeek'] = df["时间"].dt.dayofweek + 1
df = df.drop("时间区段", axis=1)
df['area'] = area_list
df.columns = ["Time", "Station", "InNum", "MorningAndEveningPeak", "OutNum", "DayOfWeek", "Area"]
df.Station = df.Station.astype('float64')

dataset_df = TSDataset.load_from_dataframe(
    df,
    group_id='Station',
    time_col='Time',  # 时间序列
    target_cols=['InNum', 'OutNum'],  # 预测目标
    freq='15min',
    static_cov_cols='Station',
    fill_missing_dates=True,
    fillna_method='pre'
)

df_train = []
df_train_test = []
df_check = []
df_test = []
for i in dataset_df:
    a, b = i.split("2023-02-28 23:45:00")
    c, d = b.split("2023-03-29 23:45:00")
    df_train.append(a)
    df_train_test.append(b)
    df_check.append(c)
    df_test.append(d)

from paddlets.transform import MinMaxScaler
from paddlets.transform import StandardScaler

min_max_scaler = StandardScaler()
df_train_scalered = min_max_scaler.fit_transform(df_train)
df_train_test_scalered = min_max_scaler.fit_transform(df_train_test)
df_check_scalered = min_max_scaler.fit_transform(df_check)
df_test_scalered = min_max_scaler.fit_transform(df_test)

paddle.seed(2023)
model = LSTNetRegressor(
    in_chunk_len=4 * 24,
    out_chunk_len=4 * 24,
    max_epochs=11,
    patience=2,
    eval_metrics=["mae"],
    optimizer_params=dict(learning_rate=3e-3)
)

model.fit(df_train_scalered, df_check_scalered)

model_save_path = "models/lstm"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
model.save(os.path.join(model_save_path, "model"))

from paddlets.utils import backtest
from paddlets.metrics import MAE

for i in df_test_scalered:
    mae, pred = backtest(data=i,
                         model=model,
                         metric=MAE(),
                         return_predicts=True
                         )
    _, ground_truth = i.split("2023-03-30 23:45:00")
    pred.plot(add_data=ground_truth)
    print(mae)
    print("\n")

test_df = pd.read_csv("data/test.csv")
station = ["A", "B", "C", "D", "E", "F", "G"]
stastionDict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def extract_station(x):
    station = re.compile("[A-Z]").findall(x)
    if len(station) > 0:
        return stastionDict[station[0]]


test_df["Station"] = test_df["Station"].apply(extract_station)
for i in station:
    dataset_test_df = test_df[test_df['Station'] == stastionDict[i]]
    dataset_test_df = TSDataset.load_from_dataframe(
        dataset_test_df,
        time_col='Time',
        target_cols=['InNum', 'OutNum'],
        freq='15min',
        fill_missing_dates=False,
    )
    StandardScaler_scaler = StandardScaler()
    ts = StandardScaler_scaler.fit_transform(dataset_test_df)
    import os
    from paddlets.models.model_loader import load

    model = load("models/lstm/model")
    res = model.predict(ts)
    res_inversed = min_max_scaler.inverse_transform(res)
    res_inversed = res_inversed.to_dataframe(copy=True)
    res_inversed["InNum"] = res_inversed["InNum"].apply(lambda x: 0 if x < 0 else int(x))
    res_inversed["OutNum"] = res_inversed["OutNum"].apply(lambda x: 0 if x < 0 else int(x))
    res_inversed["Station"] = i  # 添加站点标识
    res_inversed.index.name = "Time"
    # 输出结果文件
    if not os.path.exists("./results"):
        os.makedirs("./results")
    res_inversed.to_csv("results/{}_result.csv".format(i))

import pandas as pd

station = ["A", "B", "C", "D", "E", "F", "G"]
tt = True
for i in station:
    tempdf = pd.read_csv("results/{}_result.csv".format(i))
    if tt:
        tempdf.to_csv('预测数据量.csv', mode='a', index=False)
        tt = False
    else:
        tempdf.to_csv('预测数据量.csv', mode='a', index=False, header=False)
