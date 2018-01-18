import random
import pandas as pd
from tqdm import tqdm, trange

random.seed(13)

def lstm_dset(prediction_size, dataset): #generates a true and false sequence from real data
    df_ = {'true':[], 'false':[]}
    keys = dataset.keys()

    for i in trange(dataset.shape[0]-prediction_size):
        #true sequence
        for j in range(0, prediction_size):
            current = str(j)
            x = {}
            x[current] = {}
            for key in keys:
                x[current][key] = {}
                x[current][key] = dataset[key][i+j]
            j += 1
        df_['true'].append(x)
        #false sequence
        for j in range(0, prediction_size):
            current = str(j)
            x = {}
            x[current] = {}
            for key in keys:
                x[current][key] = {}
                pos = i + j
                if not j == 0:    #se j != 0, pega outro valor para pos, onde x[0] sempre será true
                    while i == pos: pos = random.randint(0, dataset.shape[0])
                x[current][key] = dataset[key][pos]
            j += 1
        df_['false'].append(x)

    return df_


def main():
    path = "./dataset/2013-11-03_tromso_stromsgodset_raw_first.csv"
    headers = ["timestamp", "tagid", "x_pos", "y_pos", "heading", "direction", "energy",\
    "speed", "total_distance"]

    dataset = pd.read_csv(path, names=headers)
    dataset = dataset.sort_values(by=['tagid', 'timestamp'])
    dataset = dataset.loc[:, ['tagid', 'x_pos', 'y_pos', 'heading', 'direction', 'speed']]
    dataset = pd.DataFrame(lstm_dset(1, dataset))

    print(dataset)


if __name__ == "__main__":
    main()
