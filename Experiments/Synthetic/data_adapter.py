import pandas as pd
import pathlib
from river import stream
import os
import random

path = pathlib.Path(__file__).parent.parent.resolve()
final_path  = path.joinpath(os.path.join('Synthetic', 'Data_gen'))

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
    df.to_csv(os.path.join(final_path, f'{name}.csv'), index=False)
    

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




# def handle_sine(name, data):
#     df = pd.DataFrame(data, columns=['x', 'y'])
#     df[['sine', 'cosine']] = pd.DataFrame(df['x'].tolist())
#     df.drop('x', axis=1, inplace=True)
#     df = df[['sine', 'cosine', 'y']] 
#     # data_stream = stream.iter_pandas(X=df[['sine', 'cosine']], y=df['y'])
#     anomalies = df[df['y'] == 1].shape[0]
#     df.to_csv(os.path.join(final_path, f'{name}.csv'))
#     return df.shape[0], df.shape[1]-1, anomalies


# def handle_agrawal(name, data, i):
    
#     df = pd.DataFrame(data, columns=['x', 'y'])
#     df[['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']] = pd.DataFrame(df['x'].tolist())
#     df.drop('x', axis=1, inplace=True)
#     df = df[['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan', 'y']]
#     count_y1 = df[df['y'] == 1].shape[0]
#     num_samples_to_remove = count_y1 - (250*i if (i==5 or i ==10) else 250*(i % 5))
#     df = df.drop(df[df['y'] == 1].sample(num_samples_to_remove, random_state=43).index)
#     # data_stream = stream.iter_pandas(X=df[['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']], y=df['y'])
#     anomalies = df[df['y'] == 1].shape[0]
#     # print(anomalies)
#     df.to_csv(os.path.join(final_path, f'{name}.csv'))
#     return  df.shape[0], df.shape[1]-1, anomalies


# def handle_hyper(name, data, i):
#     # df = pd.DataFrame(data, columns=['x', 'y'])
#     # sample_dict = df.loc[0, 'x']  
#     # num_features = len(sample_dict)
#     # feature_names = [str(j) for j in range(num_features)]
    
#     # df[feature_names] = pd.DataFrame(df['x'].tolist())
#     # df.drop('x', axis=1, inplace=True)
#     # df = df[feature_names + ['y']]


#     df = pd.DataFrame(data, columns=['x', 'y'])
#     # print(df) 
#     # print(f'hyper{i} \n {df.head()}')
#     df = pd.concat([df.drop('x', axis=1), df['x'].apply(pd.Series)], axis=1)
#     y = df.pop('y')
#     df['y'] = y
#     feature_names = list(df.columns)
#     feature_names.remove('y') 

#     count_y0 = df[df['y'] == 0].shape[0]
#     count_y1 = df[df['y'] == 1].shape[0]
    
#     num_samples_to_remove = count_y1 - 2*i
#     df = df.drop(df[df['y'] == 1].sample(num_samples_to_remove, random_state=42).index)
#     # data_stream = stream.iter_pandas(X=df[feature_names], y=df['y'])
#     anomalies = df[df['y'] == 1].shape[0]
#     # print(df)
#     df.to_csv(os.path.join(final_path, f'{name}.csv'), index=False)
#     return df.shape[0], df.shape[1]-1, anomalies




# def handle_tree(name, data, i):

#     df = pd.DataFrame(data, columns=['x', 'y'])
#     print(df) 
#     # print(f'hyper{i} \n {df.head()}')
#     df = pd.concat([df.drop('x', axis=1), df['x'].apply(pd.Series)], axis=1)
#     y = df.pop('y')
#     df['y'] = y
#     feature_names = list(df.columns)
#     feature_names.remove('y') 

#     count_y0 = df[df['y'] == 0].shape[0]
#     count_y1 = df[df['y'] == 1].shape[0]
    
#     num_samples_to_remove = count_y1 - 2*i
#     df = df.drop(df[df['y'] == 1].sample(num_samples_to_remove, random_state=42).index)
#     # data_stream = stream.iter_pandas(X=df[feature_names], y=df['y'])
#     anomalies = df[df['y'] == 1].shape[0]
#     print(df)
#     df.to_csv(os.path.join(final_path, f'{name}.csv'), index=False)
#     return df.shape[0], df.shape[1]-1, anomalies


# def handle_make(name, data):
#     df = pd.DataFrame(data, columns=['x', 'y'])
#     sample_dict = df.loc[0, 'x']
#     num_features = len(sample_dict)
#     feature_names = [str(j) for j in range(num_features)]
#     df[feature_names] = pd.DataFrame(df['x'].tolist())
#     df.drop('x', axis=1, inplace=True)
#     df = df[feature_names + ['y']]
#     anomalies = df[df['y'] == 1].shape[0]
#     # data_stream = stream.iter_pandas(X=df[feature_names], y=df['y'])
#     df.to_csv(os.path.join(final_path, f'{name}.csv'))
#     return df.shape[0], df.shape[1]-1, anomalies

