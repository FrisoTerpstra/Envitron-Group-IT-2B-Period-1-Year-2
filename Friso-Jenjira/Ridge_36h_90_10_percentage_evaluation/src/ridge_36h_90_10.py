#!/usr/bin/env python3
"""Envitron Prototype 2 extension — direct 36-hour Ridge forecasting.
144 outputs at 15-minute resolution; 90/10 chronological validation only.
Legacy split reproduced for comparison; see README for boundary leakage limitation.
"""
from pathlib import Path
import json, time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; DATA=ROOT/'data'; OUT=ROOT/'output'; MODELS=ROOT/'models'
TARGET='consumption_w'; BID='6a1bf87e-d798-4e93-a648-a8e948fc24c2'; STEPS=144
WEATHER=['temperature_c','relative_humidity_pct','precipitation_mm','cloud_cover_pct','shortwave_radiation_wm2','direct_radiation_wm2','diffuse_radiation_wm2','wind_speed_ms']
BASE=['origin_consumption','lag_15m','lag_1h','lag_24h','lag_7d','tod_sin','tod_cos','dow_sin','dow_cos','is_weekend','month']

def load_prepare():
    e=pd.read_csv(DATA/'electricity.csv'); w=pd.read_csv(DATA/'weather.csv'); b=pd.read_csv(DATA/'buildings.csv')
    for z in (e,w): z['timestamp_utc']=pd.to_datetime(z.timestamp_utc,utc=True,errors='coerce')
    x=e[e.building_id==BID][['building_id','timestamp_utc',TARGET]].dropna().sort_values('timestamp_utc').drop_duplicates('timestamp_utc',keep='last').copy()
    neg=x[x[TARGET]<0].copy(); x=x[x[TARGET]>=0].copy()
    wx=w[w.building_id==BID][['building_id','timestamp_utc']+WEATHER].drop_duplicates(['building_id','timestamp_utc'],keep='last')
    x=x.merge(wx,on=['building_id','timestamp_utc'],how='left').sort_values('timestamp_utc').reset_index(drop=True)
    s=x.set_index('timestamp_utc')[TARGET]
    x['origin_consumption']=x[TARGET]
    for name,delta in [('lag_15m','15min'),('lag_1h','1h'),('lag_24h','24h'),('lag_7d','7d')]: x[name]=(x.timestamp_utc-pd.Timedelta(delta)).map(s)
    mins=x.timestamp_utc.dt.hour*60+x.timestamp_utc.dt.minute; dow=x.timestamp_utc.dt.dayofweek
    x['tod_sin']=np.sin(2*np.pi*mins/1440); x['tod_cos']=np.cos(2*np.pi*mins/1440); x['dow_sin']=np.sin(2*np.pi*dow/7); x['dow_cos']=np.cos(2*np.pi*dow/7); x['is_weekend']=(dow>=5).astype(int); x['month']=x.timestamp_utc.dt.month
    # Direct multi-output targets: y_1 ... y_144 looked up by exact timestamps.
    for h in range(1,STEPS+1): x[f'y_{h}']=(x.timestamp_utc+pd.Timedelta(minutes=15*h)).map(s)
    desc=b.loc[b.building_id==BID,'description'].iloc[0] if BID in set(b.building_id) else ''
    return x,neg,desc,len(e[e.building_id==BID])

def pipe(alpha=1.0): return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('ridge',Ridge(alpha=alpha,solver='lsqr'))])

def baseline_arrays(d):
    y=d[[f'y_{h}' for h in range(1,STEPS+1)]].to_numpy(float)
    persistence=np.repeat(d.origin_consumption.to_numpy(float)[:,None],STEPS,axis=1)
    smap=d.set_index('timestamp_utc')[TARGET]
    cy=np.full_like(y,np.nan)
    for h in range(1,STEPS+1):
        target=d.timestamp_utc+pd.Timedelta(minutes=15*h)
        # Same daily slot, but never after forecast origin: 24h back for <=24h horizon,
        # 48h back for >24h horizon.
        back=1 if h<=96 else 2
        cy[:,h-1]=(target-pd.Timedelta(days=back)).map(smap).to_numpy(float)
    return y,persistence,cy

def metrics_by_horizon(validation,model,y,p):
    rows=[]
    for h in range(STEPS):
        m=np.isfinite(y[:,h])&np.isfinite(p[:,h]); yy=y[m,h]; pp=p[m,h]
        if len(yy):
            # Standard MAPE is undefined at zero actual consumption.
            positive=np.abs(yy)>=100.0  # >= 0.1 kW; explicitly report excluded points
            rows.append({'validation':validation,'model':model,'horizon_step':h+1,
                         'horizon_hours':(h+1)/4,'n':len(yy),
                         'mean_actual_w':float(np.mean(yy)),
                         'mae_w':mean_absolute_error(yy,pp),
                         'rmse_w':mean_squared_error(yy,pp)**0.5,
                         'mape_pct':float(np.mean(np.abs((yy[positive]-pp[positive])/yy[positive]))*100) if positive.any() else np.nan,
                         'mape_valid_n':int(positive.sum()),'mape_excluded_n':int((~positive).sum())})
    return rows

def run_split(d,name,train_end,test_start,test_end,save_model=False):
    tr=d[d.timestamp_utc<=train_end].copy(); te=d[(d.timestamp_utc>=test_start)&(d.timestamp_utc<=test_end)].copy()
    Ytr=tr[[f'y_{h}' for h in range(1,STEPS+1)]].to_numpy(float); Yte,pers,cy=baseline_arrays(te)
    # Require all 144 future targets for fair 36h windows.
    oktr=np.isfinite(Ytr).all(axis=1); okte=np.isfinite(Yte).all(axis=1)
    rows=[]; pred={'Persistence':pers,'Copy-Yesterday':cy}
    for n,p in pred.items(): rows += metrics_by_horizon(name,n,Yte[okte],p[okte])
    for label,features in [('Ridge',BASE),('Ridge + Weather',BASE+WEATHER)]:
        m=pipe(); m.fit(tr.loc[oktr,features],Ytr[oktr]); p=np.full_like(Yte,np.nan); p[okte]=m.predict(te.loc[okte,features]); pred[label]=p
        rows += metrics_by_horizon(name,label,Yte[okte],p[okte])
        if save_model:
            MODELS.mkdir(parents=True,exist_ok=True); joblib.dump({'model':m,'features':features,'building_id':BID,'forecast_horizon':'36 hours / 144 outputs','validation':name,'weather_mode':'forecast-origin weather only' if 'Weather' in label else 'none'},MODELS/f"{label.lower().replace(' + ','_').replace(' ','_')}_{name}.joblib")
    # selected-horizon prediction export
    pr=[]
    for h in [1,24,48,96,144]:
        idx=h-1; target_ts=te.timestamp_utc+pd.Timedelta(minutes=15*h)
        q=pd.DataFrame({'validation':name,'origin_timestamp':te.timestamp_utc,'target_timestamp':target_ts,'horizon_step':h,'horizon_hours':h/4,'actual_w':Yte[:,idx],'persistence_w':pers[:,idx],'copy_yesterday_w':cy[:,idx],'ridge_w':pred['Ridge'][:,idx],'ridge_weather_w':pred['Ridge + Weather'][:,idx]}); pr.append(q)
    return pd.DataFrame(rows),pd.concat(pr,ignore_index=True)


def main():
    t=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    (OUT/'charts').mkdir(parents=True,exist_ok=True)
    d,neg,desc,raw_n=load_prepare()
    # Legacy split is intentionally retained to reproduce previously reported 90/10 metrics.
    # This may allow training target windows to cross the test start: PRELIMINARY ONLY.
    origins=d[np.isfinite(d['y_144'])].copy()
    cut=int(len(origins)*.90)
    train_end=origins.timestamp_utc.iloc[cut-1]
    test_start=origins.timestamp_utc.iloc[cut]
    test_end=origins.timestamp_utc.iloc[-1]
    met,preds=run_split(d,'90_10',train_end,test_start,test_end,True)
    met.to_csv(OUT/'metrics_by_horizon_90_10.csv',index=False)
    preds.to_csv(OUT/'predictions_selected_horizons_90_10.csv',index=False)
    summary=met.groupby(['validation','model'],as_index=False).agg(
        mae_w=('mae_w','mean'),rmse_w=('rmse_w','mean'),mean_actual_w=('mean_actual_w','mean'),
        mape_pct=('mape_pct','mean'),mape_excluded_n=('mape_excluded_n','sum'),
        horizons=('horizon_step','nunique'))
    summary['nmae_pct']=100*summary.mae_w/summary.mean_actual_w
    summary['nrmse_pct']=100*summary.rmse_w/summary.mean_actual_w
    display=summary.copy()
    for col in ['mae_w','rmse_w','mean_actual_w']: display[col]/=1000
    display=display.rename(columns={'mae_w':'MAE (kW)','rmse_w':'RMSE (kW)',
       'mean_actual_w':'Mean actual (kW)','nmae_pct':'NMAE (%)',
       'nrmse_pct':'NRMSE (%)','mape_pct':'MAPE (%)'})
    display.to_csv(OUT/'metrics_90_10.csv',index=False)
    pd.DataFrame([{'building_id':BID,'building_description':desc,'raw_rows':raw_n,
        'negative_rows_excluded':len(neg),'negative_pct_raw':100*len(neg)/raw_n,
        'start':d.timestamp_utc.min(),'end':d.timestamp_utc.max(),
        'train_last_origin':train_end,'test_first_origin':test_start,
        'test_last_origin':test_end,'eligible_origins':len(origins)}]).to_csv(OUT/'data_quality_90_10.csv',index=False)
    # Percentage comparison across the four forecasting approaches.
    fig,ax=plt.subplots(figsize=(11,6))
    display.set_index('model')[['NMAE (%)','NRMSE (%)','MAPE (%)']].plot.bar(ax=ax)
    ax.set_title('36-hour forecast — percentage errors (90/10)')
    ax.set_ylabel('Error (%)'); ax.set_xlabel('Forecasting approach')
    ax.tick_params(axis='x',rotation=15); ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'charts/percentage_metrics_90_10.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    display.set_index('model')[['MAE (kW)','RMSE (kW)']].plot.bar(ax=ax)
    ax.set_title('36-hour forecast — absolute errors (90/10)')
    ax.set_ylabel('Error (kW)');ax.set_xlabel('Forecasting approach')
    ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'charts/model_comparison_90_10.png',dpi=180);plt.close(fig)
    for metric,unit in [('mae_w','MAE'),('rmse_w','RMSE')]:
        fig,ax=plt.subplots(figsize=(11,6))
        for label,g in met.groupby('model'):
            ax.plot(g.horizon_hours,g[metric]/1000,label=label)
        ax.set_title(f'{unit} by forecast horizon — 90/10')
        ax.set_xlabel('Horizon (hours)');ax.set_ylabel(f'{unit} (kW)');ax.legend()
        ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(OUT/f'charts/horizon_{unit.lower()}_90_10.png',dpi=180);plt.close(fig)
    v=preds[preds.horizon_step==144].dropna(subset=['actual_w','ridge_weather_w']).tail(7*96).copy()
    if len(v):
        v['target_timestamp']=pd.to_datetime(v.target_timestamp,utc=True)
        fig,ax=plt.subplots(figsize=(13,6))
        for col,label in [('actual_w','Actual'),('persistence_w','Persistence'),
                          ('copy_yesterday_w','Copy-Yesterday'),('ridge_w','Ridge'),
                          ('ridge_weather_w','Ridge + Weather')]:
            ax.plot(v.target_timestamp,v[col]/1000,label=label,alpha=.85)
        ax.set_title('Actual vs predicted — 36h horizon, 90/10 (last 7 days)')
        ax.set_ylabel('Consumption (kW)');ax.legend();ax.tick_params(axis='x',rotation=25)
        fig.tight_layout();fig.savefig(OUT/'charts/actual_vs_predicted_90_10.png',dpi=180);plt.close(fig)
    run={'experiment':'Ridge 36h 90/10 Percentage Evaluation','validation':'90/10 chronological only',
         'status':'PRELIMINARY: legacy split target-window overlap may cause leakage',
         'train_last_origin':str(train_end),'test_first_origin':str(test_start),
         'building_id':BID,'description':desc,'forecast_steps':STEPS,
         'metrics':['MAE','RMSE','NMAE','NRMSE','MAPE'],
         'weather_note':'Only weather observed at forecast origin used',
         'runtime_seconds':time.perf_counter()-t}
    (OUT/'run_summary.json').write_text(json.dumps(run,indent=2))
    print('\n90/10 chronological validation — preliminary results:')
    print(display.round(2).to_string(index=False))
    print('\nWARNING: Legacy split can allow training target overlap at test boundary.')
    print('Outputs:',OUT)

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    main()
