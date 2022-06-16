import requests
import numpy as np
import pandas as pd
import itertools
from numpy import ravel
from datetime import datetime
import math
from scipy import signal
from scipy.signal import medfilt
from scipy.signal import butter
from config import *

"Section I: download and prepare data"

def get_data(start,end,device,sensors = (SENSORS)):
    """
    sends request to server and returns
    s2 and accelerometer (nested) data only
    - start  -> starting timestamp
    - end    -> ending timestamp
    - device -> device UUID
    """
    response = requests.get(f"https://data.embry.tech/download/?start={start}&end={end}&device_UUID={device}")
    if response.status_code==200:
        data = response.json()
        data = {sensor:[[j[sensor] for j in i["doc"]["data"]] for i in data["rows"]] for sensor in sensors}
          
        assert len(data)==len(sensors)          
        return data
    else:
        print("Request error!")

def unnest_data(data):
    """
    uses itertools to unnest nested list
    """
    return list(itertools.chain.from_iterable(data))

def download_prepare_data(data, sensors=SENSORS):
    """
    First, create empty columns, for adding the data
    Second, actually download and append data
    Third, clean empty data
    Forth, create id for future merging
    """
    
    for i in sensors:
        data[f'{i}_raw']='empty'
    for i in range(0, data.shape[0]):
        try:
            data_c=get_data(*data.iloc[i,[4,5,2]], sensors = (SENSORS))
            for j in SENSORS:
                data[f'{j}_raw'].iloc[i]=unnest_data(data_c[j])
        except:
            for j in SENSORS:
                data[f'{j}_raw'].iloc[i]=[]

    data = data[data['s2_raw'].apply(lambda x: len(x)>0)].reset_index(drop=True)
    data['id']=[f"user_{data.index[i]}" for i in range(0, data.shape[0])]
    
    return data

def filter_data(data, sensors=SENSORS):
    """
    filtering data of 'x' and 'z' sensors for cycles detection in bike activities 
    """
    for i in range(0, data.shape[0]):
        data_c=get_data(*data.iloc[i,[4,5,2]], sensors = (SENSORS))
        for j in SENSORS:
                if j == 'z' or j == 'x':
                    b, a = signal.butter(4, 0.05)
                    filtered_z = signal.filtfilt(b, a, unnest_data(data_c[j]))
                    data[f'{j}_raw'].iloc[i] = filtered_z.tolist()
                else:
                    data[f'{j}_raw'].iloc[i] = unnest_data(data_c[j])
                    
    return data

def chunkify(lst, start, end, time_frame=10):
    """
    prepare data for division
    """
    bins= math.floor((end-start)/time_frame)
    if bins>0:
        n = math.floor(len(lst)/bins)
        new_list = [lst[i:i+n] for i in range(0,len(lst),n)]
        if len(lst)%math.floor((end-start)/time_frame)>0:
            new_list=new_list[:-1]
        return new_list
    else:
        return [lst]
    
def expand_data(df, column_name, time_frame):
    """
    divide data into desired timespans 
    """
    new_df=pd.DataFrame()
    id_list=[]
    new_list=[]
    for i in range(0, df.shape[0]):
        new_list.append(chunkify(df[column_name].iloc[i],*df.iloc[i,[4,5]],time_frame))
        id_list.append([df['id'].iloc[i]]*len(new_list[i]))
    new_list = list(itertools.chain.from_iterable(new_list))
    new_df=new_df.append(pd.DataFrame({column_name:new_list}))
    new_df['id']=list(itertools.chain.from_iterable(id_list))
    new_df=new_df.set_index('id')
    return new_df

"Section II: calculate features"

def median (df, sensors):
    for i in sensors:
        df[f'{i}_sorted'] = [sorted (df[f'{i}_raw'].iloc[j]) for j in range(0, df.shape[0])]
        for j in range(0, df.shape[0]):
            if len(df [f"{i}_sorted"].iloc[j])%2==0:
                k = int(len (df[f"{i}_sorted"].iloc[j])/2)
                df[f"{i}_median"] = (df[f"{i}_sorted"].iloc[j][k-1]+df[f"{i}_sorted"].iloc[j][k])/2
            else:
                k = int((len (df[f"{i}_sorted"].iloc[j])-1)/2)
                df[f"{i}_median"] = df [f"{i}_sorted"].iloc[j][k]
    return df
    
def calculate_median (df, sensors):
    for i in sensors:
        df[f'{i}_sorted'] = [sorted (df[f'{i}_raw'].iloc[j]) for j in range(0, df.shape[0])]
        for j in range(0, df.shape[0]):
            if len(df [f"{i}_sorted"].iloc[j])%2==0:
                k = int(len (df[f"{i}_sorted"].iloc[j])/2)
                df[f"{i}_median"].iloc[j] = (df[f"{i}_sorted"].iloc[j][k-1]+df[f"{i}_sorted"].iloc[j][k])/2
            else:
                k = int((len (df[f"{i}_sorted"].iloc[j])-1)/2)
                df[f"{i}_median"].iloc[j] = df [f"{i}_sorted"].iloc[j][k]
    return df    
    
def calculate_mean (df, sensors):
    for i in sensors:
        df[f'{i}_mean'] = [sum (df[f'{i}_raw'].iloc[j])/len (df[f'{i}_raw'].iloc[j]) for j in range(0, df.shape[0])]
    
    return df
    
def var (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            var_lst = [(df[f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**2 for k in range (0, len (df [f"{i}_raw"].iloc[j]))]
            df [f"{i}_var"] = sum (var_lst)/len (var_lst)
    
    return df     
    
def calculate_var (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            var_lst = [(df[f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**2 for k in range (0, len (df [f"{i}_raw"].iloc[j]))]
            df [f"{i}_var"].iloc[j] = sum (var_lst)/len (var_lst)
    
    return df    
    
def std (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_std"]= df [f"{i}_var"].iloc[j]**0.5      
    
    return df    
    
def calculate_std (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_std"].iloc[j]= df [f"{i}_var"].iloc[j]**0.5      
    
    return df    

def variation (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_variation"] = df [f"{i}_std"].iloc[j]/df[f"{i}_mean"].iloc[j]  
   
    return df
    
def calculate_variation (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_variation"].iloc[j] = df [f"{i}_std"].iloc[j]/df[f"{i}_mean"].iloc[j]  
   
    return df

def skw (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            skw_lst = [(df [f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**3 for k in range (0, len (df [f"{i}_raw"].iloc[j]))]
            df [f"{i}_skw"] = sum (skw_lst)/len (skw_lst)

    return df

def calculate_skw (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            skw_lst = [(df [f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**3 for k in range (0, len (df [f"{i}_raw"].iloc[j]))]
            df [f"{i}_skw"].iloc[j] = sum (skw_lst)/len (skw_lst)

    return df

def skew (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df[f"{i}_skew"]= df[f"{i}_skw"].iloc[j]/(df[f"{i}_std"].iloc[j]**3)

    return df

def calculate_skew (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df[f"{i}_skew"].iloc[j]= df[f"{i}_skw"].iloc[j]/(df[f"{i}_std"].iloc[j]**3)

    return df

def krt (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            krt_lst = [(df[f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**4 for k in range (0, len (df[f"{i}_raw"].iloc[j]))]
            df [f"{i}_krt"] = sum (krt_lst)/len (krt_lst)
        
    return df

def calculate_krt (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            krt_lst = [(df[f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**4 for k in range (0, len (df[f"{i}_raw"].iloc[j]))]
            df [f"{i}_krt"].iloc[j] = sum (krt_lst)/len (krt_lst)
        
    return df

def kurtosis (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_kurtosis"]= (df [f"{i}_krt"].iloc[j]/(df [f"{i}_std"].iloc[j]**4))-3

    return df

def calculate_kurtosis (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_kurtosis"].iloc[j]= (df [f"{i}_krt"].iloc[j]/(df [f"{i}_std"].iloc[j]**4))-3

    return df

def tvar (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            var_lst = [(df [f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**2 for k in range (0, len (df[f"{i}_raw"].iloc[j]))]
            df [f"{i}_tvar"] = sum (var_lst)/(len (var_lst) -1) 
 
    return df

def calculate_tvar (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            var_lst = [(df [f"{i}_raw"].iloc[j][k]-df[f"{i}_mean"].iloc[j])**2 for k in range (0, len (df[f"{i}_raw"].iloc[j]))]
            df [f"{i}_tvar"].iloc[j] = sum (var_lst)/(len (var_lst) -1) 
 
    return df

def kstat(data, n=2):
    if n > 4 or n < 1:
        raise ValueError("k-statistics only supported for 1<=n<=4")
    n = int(n)
    S = np.zeros(n + 1, np.float64)
    data = ravel(data)
    N = data.size
    if N == 0:
        raise ValueError("Data input must not be empty")

    # on nan input, return nan without warning
    if np.isnan(np.sum(data)):
        return np.nan

    for k in range(1, n + 1):
        S[k] = np.sum(data**k, axis=0)
    if n == 1:
        return S[1] * 1.0/N
    elif n == 2:
        return (N*S[2] - S[1]**2.0) / (N*(N - 1.0))
    elif n == 3:
        return (2*S[1]**3 - 3*N*S[1]*S[2] + N*N*S[3]) / (N*(N - 1.0)*(N - 2.0))
    elif n == 4:
        return ((-6*S[1]**4 + 12*N*S[1]**2 * S[2] - 3*N*(N-1.0)*S[2]**2 -
                 4*N*(N+1)*S[1]*S[3] + N*N*(N+1)*S[4]) /
                 (N*(N-1.0)*(N-2.0)*(N-3.0)))
    else:
        raise ValueError("Should not be here.")
        
def kstatvar(data, n=2):
    data = ravel(data)
    N = len(data)
    if n == 1:
        return kstat(data, n=2) * 1.0/N
    elif n == 2:
        k2 = kstat(data, n=2)
        k4 = kstat(data, n=4)
        return (2*N*k2**2 + (N-1)*k4) / (N*(N+1))
    else:
        raise ValueError("Only n=1 or n=2 supported.")
        
def calculate_kstat (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_kstatvar"] = kstatvar (df [f"{i}_raw"].iloc[j])
            
    return df

def calculate_kstatvar (df, sensors):
    for i in sensors:
        for j in range(0, df.shape[0]):
            df [f"{i}_kstatvar"].iloc[j] = kstatvar (df [f"{i}_raw"].iloc[j])
            
    return df

    
