import pickle


def get_scatter_data():
    return pickle.load(open("data_sample/scatter_data.pkl", "rb"))


def get_scatter_data_3D():
    return pickle.load(open("data_sample/scatter_data_3D.pkl", "rb"))

