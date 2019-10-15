import queue

import numpy as np
from sklearn.cluster import KMeans


class Event:
    def __init__(self, begin, end, description, v):
        self.b = begin
        self.e = end
        self.description = description
        self.v = v

    def update_description(self, description):
        self.description = description

    def get_minutes(self):
        return (self.e - self.b).days * 24 * 60 + (self.e - self.b).seconds / 60

    def print(self):
        print("时间：%s - %s；事件：%s 用时 %.2f 小时" %
              (self.b, self.e, self.description, self.get_minutes() / 60))


def get_centers(sample, column):
    cluster = []

    for i in range(len(sample)):
        cluster.append(sample.iloc[i][column])

    cluster = np.array(cluster).reshape((-1, 1))

    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(cluster)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心
    centers = centroids.reshape(-1)
    centers.sort()
    return centers


def get_work_events(df, threshold, ref_column, ignore_time, break_time):
    # FIXME 降低复杂性到 O(n)
    tags = []
    events = []
    for i in range(len(df)):
        if df.iloc[i][ref_column] > threshold:
            tags.append(1)
        else:
            tags.append(0)

    e = Event(df.iloc[0]["datetime"], -1, -1, tags[0])
    i = 1
    while True:
        if i >= len(df):
            break
        if tags[i] != e.v or i == len(df) - 1:
            e.e = df.iloc[i - 1]["datetime"] + \
                  (df.iloc[i]["datetime"] - df.iloc[i - 1]["datetime"]) / 2
            if e.v == 1:
                e.update_description("开机")
            else:
                if e.get_minutes() < break_time:
                    e.update_description("休息")
                else:
                    e.update_description("停机")
            events.append(e)
            e = Event(e.e, -1, -1, tags[i])
        i += 1

    find = True
    new_events = []
    while find:
        find = False
        new_events.append(events[0])
        for i in range(1, len(events) - 1):
            if events[i].get_minutes() < ignore_time:
                find = True
                new_events[-1].e = events[i + 1].e
                end = i + 1
                break
            else:
                new_events.append(events[i])
        if find:
            for i in range(end + 1, len(events)):
                new_events.append(events[i])
        else:
            new_events.append(events[-1])

        events, new_events = new_events, []

    return tags, events


class EnergyStatusCluster:
    def __init__(self, data_df, column, threshold, ignore_time, break_time):
        self.df = data_df
        self.column = column
        self.threshold = threshold
        self.ignore_time = ignore_time
        self.break_time = break_time
        self.event_queue = []
        self.begin_index, self.end_index = None, None
        # self.event_queue = queue.Queue()
        return

    def analyze(self):
        centers = get_centers(self.df, self.column)

        threshold = (1 - self.threshold) * centers[0] + self.threshold * centers[1]

        tags, es = get_work_events(self.df, threshold, self.column, self.ignore_time, self.break_time)
        self.df["tag"] = tags
        self.event_queue = es

    def get_energy_time(self):
        # for e in self.event_queue:
        #     e.print()
        ret = []
        q = self.event_queue
        begin_time, end_time = None, None
        for i in range(1, len(q)):
            if begin_time is None:
                if q[i - 1].description == "停机" and q[i].description == "开机":
                    begin_time = q[i].b
                    self.begin_index = i
            else:
                if end_time is not None:
                    break
                if q[i - 1].description == "开机" and q[i].description == "停机":
                    end_time = q[i].b
                    self.end_index = i

        return [begin_time, end_time]

    def get_break_info(self):
        if self.begin_index is None:
            self.get_energy_time()
        nums, time = 0, 0
        if self.begin_index is None:
            return [0, 0]
        if self.end_index is None:
            self.end_index = len(self.event_queue)
        for i in range(self.begin_index, self.end_index):
            if self.event_queue[i].description == "休息":
                nums += 1
                time += self.event_queue[i].get_minutes()

        return [nums, time / 60]
