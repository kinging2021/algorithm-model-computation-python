import json
import datetime
import requests

import pandas as pd

"""
调用方式: get_jinmailang_df(begin_time, end_time)
返回 DataFrame
"""


def get_history_data(begin_time, days=2):
    begin_s = datetime.datetime.strftime(begin_time, "%Y-%m-%d %H:00:00")
    end_time = begin_time + datetime.timedelta(days=days)
    end_s = datetime.datetime.strftime(end_time, "%Y-%m-%d 00:00:00")
    url = "https://edge.fanneng.com/jml//ems/device/findLineNowDH/PARK120_EMS05/14472/%s/%s/5/1"
    url = url % (begin_s, end_s)
    # print(url)
    r = requests.get(url, verify=False)
    try:
        data = r.json()
        return data
    except:
        raise Exception("error occurred when getting data from %s" % url)


def get_jml_df(json_obj):
    df_dict = {}

    def load_column(x_str, y_str, name):
        x = [datetime.datetime.strptime(v, "%Y-%m-%d %H:%M:%S") for v in x_str]
        y = [0 if v is None or v == "" else float(v) for v in y_str]
        df_dict["datetime"] = x
        df_dict[name] = y

    for i in range(len(json_obj['data'])):
        load_column(json_obj['data'][i]['xvalue'], json_obj['data']
        [i]['lineValue'], json_obj['data'][i]["name"])

    df = pd.DataFrame(df_dict)
    return df


def add_features(df):
    def get_time_feature(df):
        def get_time(row):
            return row

        def get_year(row):
            time = get_time(row)
            return time.year

        def get_month(row):
            time = get_time(row)
            return time.month

        def get_day(row):
            time = get_time(row)
            return time.day

        def get_weekday(row):
            time = get_time(row)
            return time.weekday()

        def get_hour(row):
            time = get_time(row)
            return time.hour

        def get_minute(row):
            time = get_time(row)
            return time.minute

        def get_second(row):
            time = get_time(row)
            return time.second

        if len(df) <= 0:
            print("warning: lenght 0")
            return
        if "datetime" in df.columns:
            df["year"] = df["datetime"].apply(get_year)
            df["month"] = df["datetime"].apply(get_month)
            df["day"] = df["datetime"].apply(get_day)
            df["weekday"] = df["datetime"].apply(get_weekday)
            df["hour"] = df["datetime"].apply(get_hour)
            df["minute"] = df["datetime"].apply(get_minute)
            df["second"] = df["datetime"].apply(get_second)
        return

    df["电"] = df["和面（电）"] + df["压延（电）"] + df["油炸（电）"] + df["包装（电）"]
    df["蒸汽"] = df["蒸箱(蒸汽)"] + df["油炸(蒸汽)"]
    get_time_feature(df)
    df["hour_minute"] = df["hour"] + df["minute"] / 60
    return df


def get_jinmailang_df(begin_time, days=2):
    data = get_history_data(begin_time, days)
    df = get_jml_df(data)
    df = add_features(df)
    return df.iloc[:-1]
