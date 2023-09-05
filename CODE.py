# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:55:54 2022

@author: HP
"""

import pandas as pd
x=pd.read_csv("D:\\New data\\SENSEX MAIN\\OUTPUT.csv")
df= pd.DataFrame(x)

Y = df['sensex_open']
X=df[['asp_close','asp_value','ab_close','ab_value','bf_close','bf_value','bfv_close','bfv_value','ba_close','ba_value','dr_close','dr_value','hcl_close','hcl_value','hdfc_close','hdfc_value','hdb_close','hdb_value','hul_close','hul_value','icici_close','icici_value','ib_close','ib_value','ifs_close','ifs_value','itc_close','itc_value','kmb_close','kmb_value','lt_close','lt_value','mm_close','mm_value','maruthi_close','maruthi_value','nestle_close','nestle_value','ntpc_close','ntpc_value','pg_close','pg_value','ri_close','ri_value','sbi_close','sbi_value','sun_close','sun_value','ts_close','ts_value','tcs_close','tcs_value','tm_close','tm_value','titan_close','titan_value','uc_close','uc_value','wipro_close','wipro_value']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression


#using Stat model
from statsmodels.api import OLS
model_lin=OLS.from_formula("sensex_open~asp_close+asp_value+ab_close+ab_value+bf_close+bf_value+bfv_close+bfv_value+ba_close+ba_value+dr_close+dr_value+hcl_close+hcl_value+hdfc_close+hdfc_value+hdb_close+hdb_value+hul_close+hul_value+icici_close+icici_value+ib_close+ib_value+ifs_close+ifs_value+itc_close+itc_value+kmb_close+kmb_value+lt_close+lt_value+mm_close+mm_value+maruthi_close+maruthi_value+nestle_close+nestle_value+ntpc_close+ntpc_value+pg_close+pg_value+ri_close+ri_value+sbi_close+sbi_value+sun_close+sun_value+ts_close+ts_value+tcs_close+tcs_value+tm_close+tm_value+titan_close+titan_value+uc_close+uc_value+wipro_close+wipro_value",data=df)

result_lin = model_lin.fit()
result_lin.summary()


lm = LinearRegression(copy_X=True, fit_intercept=False)

lm.fit(X_train,Y_train)
print(lm.intercept_)
print(lm.coef_)
print(lm.score(X_train,Y_train))
