'''
Utiliy function for the ResNet50-CoxRegression network

'''
import numpy as np 
import pandas as pd 

def contains(target_str, search_arr):
    res = False

    for search_str in search_arr:
        if search_str in target_str:
            res = True
            break

    return res

def make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set

def median_cindex_data(risk=None,time= None,cens= None,patient_ID = None):
    assert len(risk) == time.shape[0]
    df = {'patient_ID':patient_ID,'time':time,'cens':cens,'risk':risk}
    df = pd.DataFrame(df)
#     df = df[:len(risk)]
#     df['risk'] = risk
    df_reduced = df.groupby('patient_ID').aggregate(np.median)#df.loc[df.risk == risk_median]
    return df_reduced
