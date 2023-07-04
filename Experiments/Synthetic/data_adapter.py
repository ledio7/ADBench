import pandas as pd
import pathlib
from river import stream
import os
import random

cols ={
    "hyper1": 50, "hyper2":100, "hyper3":200, "hyper4":300, "hyper5":400, 
    "tree1":50, "tree2":100, "tree3":150, "tree4":200, "tree5":250,
    "make1":200, "make2":200, "make3":300, "make4":400, "make5":400
}


def handle_dataset(name, data):
    df = pd.DataFrame(data, columns=['x', 'y'])
    # print(df) 
    # print(f'hyper{i} \n {df.head()}')
    df = pd.concat([df.drop('x', axis=1), df['x'].apply(pd.Series)], axis=1)
    y = df.pop('y')
    df['y'] = y
    feature_names = list(df.columns)
    feature_names.remove('y') 

    count_y0 = df[df['y'] == 0].shape[0]
    count_y1 = df[df['y'] == 1].shape[0]
    
    if 'sine' not in name and 'make' not in name:
        i = int(''.join(filter(str.isdigit, name)))
        num_samples_to_remove = count_y1 - 250*i
        df = df.drop(df[df['y'] == 1].sample(num_samples_to_remove, random_state=42).index)
    anomalies = df[df['y'] == 1].shape[0]
    # df = df.sample(frac=1)
    # print(df)
    print(anomalies)
    df.to_csv(os.path.join('Data_gen', f'{name}.csv'), index=False)
    

def get_converters(name):
    # print(cols)
    if 'sine' in name: 
        converters = {"sine": float, "cosine": float}
    elif 'agra' in name:
        converters = {"salary": float, "commission": float, "age": float, "elevel": float, "car": float, "zipcode": float, "hvalue": float, "hyears": float, "loan": float}
    elif 'tree' in name:
        converters = {f"x_num_{i}": float for i in range(0, cols[name])}
    else:
        converters = {f"{i}": float for i in range(0, cols[name])}
    converters["y"] = float
    return converters

