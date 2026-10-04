#!/usr/bin/env python3
"""Envitron Prototype 2 - Ridge: lightweight, explainable 15-minute consumption forecasting.

Compares Persistence, Copy-Yesterday, Ridge, and Ridge + Weather for one building.
Raw client data are never modified. 
Negative consumption values are flagged and preserved.
"""
from pathlib import Path
import argparse, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
MODELS = ROOT / "models"
WEATHER=['temperature_c','relative_humidity_pct','precipitation_mm','cloud_cover_pct','shortwave_radiation_wm2','direct_radiation_wm2','diffuse_radiation_wm2','wind_speed_ms']
BASE_FEATURES=['lag_15m','lag_30m','lag_1h','lag_24h','lag_7d','tod_sin','tod_cos','dow_sin','dow_cos','is_weekend','month']
TARGET='consumption_w'

def lookup_lag(s, timestamps, delta):
    return (timestamps-delta).map(s)

def quality(e, w, bid):
    x=e[e.building_id==bid].copy(); x['timestamp_utc']=pd.to_datetime(x.timestamp_utc,utc=True,errors='coerce'); x=x.sort_values('timestamp_utc')
    dif=x.timestamp_utc.diff().dropna(); dominant=dif.mode().iloc[0] if len(dif) else pd.NaT
    wx=w[w.building_id==bid].copy(); wx['timestamp_utc']=pd.to_datetime(wx.timestamp_utc,utc=True,errors='coerce')
    return pd.DataFrame([{'building_id':bid,'start':x.timestamp_utc.min(),'end':x.timestamp_utc.max(),'observations':len(x),'missing_consumption':int(x[TARGET].isna().sum()),'duplicate_timestamps':int(x.timestamp_utc.duplicated().sum()),'negative_consumption':int((x[TARGET]<0).sum()),'negative_pct':float((x[TARGET]<0).mean()*100),'dominant_interval':str(dominant),'weather_rows':len(wx)}])

def prepare(e,w,bid):
    x=e[e.building_id==bid][['building_id','timestamp_utc',TARGET]].copy(); x['timestamp_utc']=pd.to_datetime(x.timestamp_utc,utc=True,errors='coerce'); x=x.dropna(subset=['timestamp_utc']).sort_values('timestamp_utc').drop_duplicates('timestamp_utc',keep='last')
    wx=w[w.building_id==bid].copy(); wx['timestamp_utc']=pd.to_datetime(wx.timestamp_utc,utc=True,errors='coerce'); wx=wx.drop_duplicates(['building_id','timestamp_utc'],keep='last')
    x=x.merge(wx[['building_id','timestamp_utc']+WEATHER],on=['building_id','timestamp_utc'],how='left')
    s=x.set_index('timestamp_utc')[TARGET]
    for name,delta in [('lag_15m','15min'),('lag_30m','30min'),('lag_1h','1h'),('lag_24h','24h'),('lag_7d','7d')]: x[name]=lookup_lag(s,x.timestamp_utc,pd.Timedelta(delta)).to_numpy()
    minutes=x.timestamp_utc.dt.hour*60+x.timestamp_utc.dt.minute; dow=x.timestamp_utc.dt.dayofweek
    x['tod_sin']=np.sin(2*np.pi*minutes/1440); x['tod_cos']=np.cos(2*np.pi*minutes/1440); x['dow_sin']=np.sin(2*np.pi*dow/7); x['dow_cos']=np.cos(2*np.pi*dow/7); x['is_weekend']=(dow>=5).astype(int); x['month']=x.timestamp_utc.dt.month
    # Baselines are timestamp-based, not row-offset based.
    x['persistence']=x['lag_15m']; x['copy_yesterday']=x['lag_24h']
    return x

def pipe(features, alpha):
    return Pipeline([('prep',ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),features)],remainder='drop')),('ridge',Ridge(alpha=alpha))])

def metrics(name,y,p,train_s=0,pred_s=0,size=0):
    mask=np.isfinite(y)&np.isfinite(p); yy=np.asarray(y)[mask]; pp=np.asarray(p)[mask]
    return {'model':name,'n_evaluated':len(yy),'mae_w':mean_absolute_error(yy,pp),'rmse_w':mean_squared_error(yy,pp)**0.5,'training_seconds':train_s,'prediction_seconds':pred_s,'model_size_bytes':size}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--building-id'); ap.add_argument('--alpha',type=float,default=1.0); ap.add_argument('--test-fraction',type=float,default=.20); ap.add_argument('--plot-days',type=int,default=7); a=ap.parse_args()
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    e=pd.read_csv(DATA/'electricity.csv'); w=pd.read_csv(DATA/'weather.csv'); b=pd.read_csv(DATA/'buildings.csv')
    bid=a.building_id or b.loc[b.description.str.contains('office',case=False,na=False),'building_id'].iloc[0]
    if bid not in set(e.building_id): raise SystemExit('Unknown building-id')
    q=quality(e,w,bid); q.to_csv(OUT/'data_quality.csv',index=False)
    d=prepare(e,w,bid); usable=d.dropna(subset=[TARGET]+BASE_FEATURES).copy()
    if len(usable)<1000: raise SystemExit('Not enough usable observations.')
    cut=int(len(usable)*(1-a.test_fraction)); train=usable.iloc[:cut].copy(); test=usable.iloc[cut:].copy()
    results=[]; preds=test[['building_id','timestamp_utc',TARGET,'persistence','copy_yesterday']].copy()
    results.append(metrics('Persistence',test[TARGET],test.persistence)); results.append(metrics('Copy-Yesterday',test[TARGET],test.copy_yesterday))
    fitted={}
    for label,features,filename in [('Ridge',BASE_FEATURES,'ridge.joblib'),('Ridge + Weather',BASE_FEATURES+WEATHER,'ridge_weather.joblib')]:
        model=pipe(features,a.alpha); t=time.perf_counter(); model.fit(train[features],train[TARGET]); train_s=time.perf_counter()-t
        t=time.perf_counter(); p=model.predict(test[features]); pred_s=time.perf_counter()-t; preds[label.lower().replace(' + ','_').replace(' ','_')]=p
        path=MODELS/filename; joblib.dump({'model':model,'features':features,'building_id':bid,'target':TARGET,'forecast_horizon':'15 minutes','alpha':a.alpha},path); results.append(metrics(label,test[TARGET],p,train_s,pred_s,path.stat().st_size)); fitted[label]=(model,features)
    pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False); preds.to_csv(OUT/'predictions.csv',index=False)
    coef=[]
    for label,(model,features) in fitted.items():
        vals=model.named_steps['ridge'].coef_
        for f,c in zip(features,vals): coef.append({'model':label,'feature':f,'standardized_coefficient':c,'abs_coefficient':abs(c)})
    cdf=pd.DataFrame(coef).sort_values(['model','abs_coefficient'],ascending=[True,False]); cdf.to_csv(OUT/'coefficients.csv',index=False)
    # Charts
    view=preds.tail(a.plot_days*96)
    plt.figure(figsize=(13,6)); plt.plot(view.timestamp_utc,view[TARGET],label='Actual'); plt.plot(view.timestamp_utc,view.persistence,label='Persistence',alpha=.75); plt.plot(view.timestamp_utc,view.copy_yesterday,label='Copy-Yesterday',alpha=.75); plt.plot(view.timestamp_utc,view.ridge,label='Ridge',alpha=.85); plt.legend(); plt.title('Actual vs Forecast — Representative Recent Period'); plt.ylabel('Consumption (W)'); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(OUT/'forecast_comparison.png',dpi=180); plt.close()
    plt.figure(figsize=(13,5)); plt.plot(view.timestamp_utc,abs(view[TARGET]-view.copy_yesterday),label='Copy-Yesterday absolute error'); plt.plot(view.timestamp_utc,abs(view[TARGET]-view.ridge),label='Ridge absolute error'); plt.legend(); plt.title('Absolute Forecast Error'); plt.ylabel('Absolute error (W)'); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(OUT/'error_comparison.png',dpi=180); plt.close()
    m=pd.DataFrame(results); plt.figure(figsize=(8,5)); plt.bar(m.model,m.mae_w); plt.title('Model MAE Comparison'); plt.ylabel('MAE (W)'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(OUT/'model_mae.png',dpi=180); plt.close()
    cr=cdf[cdf.model=='Ridge'].sort_values('standardized_coefficient'); plt.figure(figsize=(9,6)); plt.barh(cr.feature,cr.standardized_coefficient); plt.title('Ridge Standardized Coefficients'); plt.xlabel('Coefficient'); plt.tight_layout(); plt.savefig(OUT/'ridge_coefficients.png',dpi=180); plt.close()
    summary={'building_id':bid,'description':b.loc[b.building_id==bid,'description'].iloc[0] if bid in set(b.building_id) else '', 'target':TARGET,'forecast_horizon':'15 minutes (experimental assumption)','train_rows':len(train),'test_rows':len(test),'alpha':a.alpha,'note':'Negative consumption values are preserved pending client clarification.'}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('\nPrototype 2 complete.\n',pd.DataFrame(results).to_string(index=False)); print('\nOutputs:',OUT); print('Models:',MODELS)
if __name__=='__main__': main()
