import pandas as pd
import numpy as np
import seaborn as sns
from acquire import get_zillow_data
from sklearn.model_selection import train_test_split

import scipy.stats as scs
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

PATH = "/content/drive/MyDrive/Colab Notebooks/clustering/data/"


def wrangle_zillow(test_size, clip=True, thresh=.4, random_state=0, path=PATH):
    
    #df = get_zillow_data()
    #df.to_csv(path+"sup_2017.csv", index=False)
    df = pd.read_csv(path+"sup_2017.csv", index_col='parcelid')
    df = df.rename(columns={'calculatedfinishedsquarefeet':'finishedsqft',
                            'lotsizesquarefeet':'lotsqft',
                            'structuretaxvaluedollarcnt':'structuretaxvalue',
                            'taxvaluedollarcnt':'taxvalue',
                            'landtaxvaluedollarcnt':'landtaxvalue',
                            'buildingqualitytypeid' : 'buildquality',
                            'heatingorsystemtypeid' : 'heating',
                            'taxdelinquencyflag': 'delinquet'})
                            
    df.delinquet.replace("Y", 1, inplace=True)
    df.heating = df.heating.map({20: 'Solar', 6: 'Forced air', 7: 'Floor', 2: 'Central'})
    
    df = zero_fill(df, ['delinquet','poolcnt', 'heating', 'regionidcity'])
    df = median_fill(df, ['buildquality', 'lotsqft'])
    
    df = drop_columns(df, thresh)
    print(f"\n{df.isna().any(axis=1).sum()} incomplete cases dropped from the data.")
    df = df.dropna()
    
    df.latitude, df.longitude = df.latitude/1e6, df.longitude/1e6
    df['livingarearatio'] = df.finishedsqft/df.lotsqft
    df['buildinglandvalueratio'] = df.structuretaxvalue/df.landtaxvalue 
    df['taxrate'] = df.taxamount/df.taxvalue
    df['age'] = 2017-df.yearbuilt
    df['age_bins'] = pd.qcut(df.age, 2, labels=[0,1]) #newer == 0
    df['acres'] = df.lotsqft/43560
    df['sqftvalue'] = df.structuretaxvalue/df.finishedsqft
    df['landsqftvalue'] = df.landtaxvalue/df.lotsqft
    
    if clip==True:
        df = percentile_method(df, ["logerror"])
    
    zips = pd.get_dummies(df.regionidzip).drop(columns=zip_ttests(df))
    df.fips = df.fips.map({6111:"Ventura", 6037:"Los Angeles", 6059:"Orange"})
    df = pd.concat([df, pd.get_dummies(df.fips),
                    pd.get_dummies(df.heating), zips], axis=1)
    df = df.drop(columns=["fips", "heating", "regionidzip", "yearbuilt", "lotsqft", 0])
    
    return split_data(df, 'logerror', test_size, random_state)


def zip_ttests(X, alpha=.05):
    
    mean_error = X.logerror.mean()
    zips = set(X.regionidzip)
    insig_zips = []
    
    for zip_ in zips:
        
        subset = X.logerror[X.regionidzip==zip_]
        mean_vec = np.zeros((1, len(subset))) + mean_error
        
        #Welch's T-test comparing zip code logerror mean to pop. mean
        p = scs.ttest_ind(subset, mean_vec[0], equal_var=False)[1]
        if p > alpha: insig_zips.append(zip_)
        
    return insig_zips
    
    
def percentile_method(df, cols):
    
    df_len = len(df)
    
    for col in cols:
        
        ulimit = np.percentile(df[col], 99)
        llimit = np.percentile(df[col], 1)
        #df[col][df[col]>ulimit] = ulimit
        #df[col][df[col]<llimit] = llimit
        df = df[df[col]<ulimit]
        df = df[df[col]>llimit]
    
    print(f"{df_len - len(df)} outliers clipped at 1st and 99th percentiles based on {cols}.")
    return df


def split_data(X, target, test_size, random_state):
    
    y = X.pop(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    test_size2 = test_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size2, random_state=random_state)
    print(f"\nX_train {X_train.shape}, X_test {X_test.shape}, X_val {X_val.shape}")
    print(f"y_train {y_train.shape}, y_test {y_test.shape}, y_val {y_val.shape}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def drop_columns(df, thresh):
    
    #drop columns missing more than threshold percentage
    missing = df.isna().sum()
    out = pd.DataFrame(pd.concat([missing, missing/len(df)], axis=1))
    out = out.rename(columns={0:'missing count', 1:'missing %'})
    out = out[out['missing %'] > thresh]
    print("\nThese columns were dropped.")
    print(out)
    
    return df.drop(columns=list(out.index))


def zero_fill(df, cols):
    
    print(f"\nMissing values in {cols} replaced with zeros")
    for col in cols:
        df[col].fillna(0, inplace=True)
        
    return df
    

def median_fill(df, cols):
    
    print(f"\nMissing values in {cols} replaced with median")
    for col in cols:
        df[col].fillna(np.nanmedian(df[col]), inplace=True)
        
    return df


def outlying_X(df, k):
    
    features=dict(livingarearatio=[], buildinglandvalueratio=[], age=[],
                  sqftvalue=[], landsqftvalue=[], taxvalue=[], finishedsqft=[],
                  acres=[], taxrate=[])
    df['outlying_X'] = 0
    
    for col in features:
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper, lower = q3 + k*iqr, q1 - k*iqr
        features[col] = [lower, upper]
    
    for i in range(len(df)):
        j = 0
        for col in features:
            v = df[col].iloc[i]
            lo, hi = features[col][0], features[col][1]
            if v > hi*k or v < lo*k:
                j += 1
        df.outlying_X.iloc[i] = j
        
    return df
    
    
def plot_corr(X):
    #A lower-triangle correlation heatmap
    plt.figure(figsize=(20,9))
    corr = X.corr()
    return sns.heatmap(corr, mask=np.triu(corr), annot=True)


def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [round(variance_inflation_factor(X.values, i),1) for i in range(X.shape[1])]
    return(vif)