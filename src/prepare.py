import pandas as pd
import numpy as np
import seaborn as sns
from acquire import get_zillow_data
from sklearn.model_selection import train_test_split

import scipy.stats as scs
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

PATH = "/content/drive/MyDrive/Colab Notebooks/clustering/data/"


def wrangle_zillow(test_size, k=2.5, thresh=.4, random_state=0, path=PATH):
    
    #df = get_zillow_data()
    #df.to_csv(path+"sup_2017.csv", index=False)
    df = pd.read_csv(path+"sup_2017.csv", index_col='parcelid')
    df = df.rename(columns={'calculatedfinishedsquarefeet':'finishedsqft',
                            'lotsizesquarefeet':'lotsqft',
                            'structuretaxvaluedollarcnt':'structuretaxvalue',
                            'taxvaluedollarcnt':'taxvalue',
                            'landtaxvaluedollarcnt':'landtaxvalue',
                            'buildingqualitytypeid' : 'buildquality'})
    
    df.poolcnt.fillna(0, inplace=True)
    df = median_fill(df, ['buildquality', 'lotsqft'])
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
    
    #df = iqr_method(df, k, ["taxrate", "taxvalue"])
    #rather than drop outliers, I featurized them
    df = outlying_X(df, k)
    
    zips = pd.get_dummies(df.regionidzip).drop(columns=zip_ttests(df))
    df.fips = df.fips.map({6111:"Ventura", 6037:"Los Angeles", 6059:"Orange"})
    df = pd.concat([df, pd.get_dummies(df.fips), zips], axis=1)
    df = df.drop(columns=["fips", "regionidzip", "yearbuilt", "lotsqft",
                            "taxamount", "landtaxvalue", "structuretaxvalue"])
    
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
    
    
def iqr_method(df, k, cols):
    
    was = len(df)
    #drop row if column outside fences, k is usu. between 1.5 and 3
    for col in cols:
        
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upperbound, lowerbound = q3 + k*iqr, q1 - k*iqr
        #bool_mask = (df[col] < lowerbound) | (df[col] > upperbound)
        df = df[(df[col] > lowerbound) & (df[col] < upperbound)]
    
    print(f"\n{was-len(df)} outliers dropped based on {cols}.")
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


def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    """Mean absolute percentage error regression loss.
    Note here that we do not represent the output as a percentage in range
    [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
    :ref:`User Guide <mean_absolute_percentage_error>`.
    .. versionadded:: 0.24
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats in the range [0, 1/eps]
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.
    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_percentage_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.3273...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.5515...
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.6198...
    """
    
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    """
    
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)