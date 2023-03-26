from sklearn import linear_model, preprocessing
import numpy as np
import pandas as pd

def getAttributes(df, class_atr = "class"):
    if class_atr not in df.columns:
        raise Exception("Class attribute not in dataframe")
    attrs = []  
    for col in df.columns:
        if col != class_atr:
            attrs.append(col)
    return attrs

def ColumnsToList(df, columns, class_atr = "class"):
    list_of_values = []
    for col in columns:
        try:
            list_of_values.append(df[col].tolist())
        except:
            raise Exception("Column not in dataframe")

    try:
        class_attrs = df[class_atr].tolist()
    except:
        raise Exception("Class attribute not in dataframe")
    
    return list_of_values, class_attrs



def preprocessData(df):
    le = preprocessing.LabelEncoder()
    attrs = getAttributes(df)
    cols_values, class_values = ColumnsToList(df, attrs)
    preprocess_lists = []
    dict_attrs_values = {}
    for idx, col in enumerate(cols_values):
        preprocess_lists.append(le.fit_transform(col))
        dict_attrs_values[attrs[idx]] = dict(zip(range(len(le.classes_)), le.classes_))
    

    cls_fit = le.fit(class_values)
    cls_ = cls_fit.transform(class_values)
    values_map = dict(zip(range(len(le.classes_)), le.classes_))

    return preprocess_lists, cls_, values_map, dict_attrs_values

def getXandY(df):
    x = list(zip(*preprocessData(df)[0]))
    y = list(preprocessData(df)[1])
    return x, y


    
    