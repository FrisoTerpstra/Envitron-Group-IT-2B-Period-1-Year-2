#!/usr/bin/env python3
"""Envitron Prototype 2 extension — direct 36-hour Ridge forecasting.
144 outputs at 15-minute resolution; chronological 80/20, 90/10 and walk-forward validation.
"""
from pathlib import Path
import json, time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; DATA=ROOT/'data'; OUT=ROOT/'output'/'experiment_36h'; MODELS=ROOT/'models'/'experiment_36h'
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
        if len(yy): rows.append({'validation':validation,'model':model,'horizon_step':h+1,'horizon_hours':(h+1)/4,'n':len(yy),'mae_w':mean_absolute_error(yy,pp),'rmse_w':mean_squared_error(yy,pp)**0.5})
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
    t=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    d,neg,desc,raw_n=load_prepare(); neg.to_csv(OUT/'excluded_negative_consumption.csv',index=False)
    pd.DataFrame([{'building_id':BID,'description':desc,'raw_rows':raw_n,'negative_rows_excluded':len(neg),'negative_pct_raw':len(neg)/raw_n*100,'rows_after_exclusion':len(d),'start':d.timestamp_utc.min(),'end':d.timestamp_utc.max()}]).to_csv(OUT/'data_quality.csv',index=False)
    # Origins with complete 36h target window.
    origins=d[np.isfinite(d['y_144'])].copy(); n=len(origins); ms=[]; ps=[]
    for frac,name in [(0.20,'80_20'),(0.10,'90_10')]:
        cut=int(n*(1-frac)); m,p=run_split(d,name,origins.timestamp_utc.iloc[cut-1],origins.timestamp_utc.iloc[cut],origins.timestamp_utc.iloc[-1],True); ms.append(m); ps.append(p)
    # 3 expanding walk-forward folds across the latter 40% of origins.
    bounds=np.linspace(int(n*.60),n,4,dtype=int)
    for i in range(3):
        a,z=bounds[i],bounds[i+1]; m,_=run_split(d,f'walk_forward_{i+1}',origins.timestamp_utc.iloc[a-1],origins.timestamp_utc.iloc[a],origins.timestamp_utc.iloc[z-1]); ms.append(m)
    met=pd.concat(ms,ignore_index=True); preds=pd.concat(ps,ignore_index=True); met.to_csv(OUT/'metrics_by_horizon.csv',index=False); preds.to_csv(OUT/'predictions_selected_horizons.csv',index=False)
    # Summary: mean horizon metrics; walk-forward combined across folds.
    summary=met[~met.validation.str.startswith('walk_forward_')].groupby(['validation','model'],as_index=False).agg(mae_w=('mae_w','mean'),rmse_w=('rmse_w','mean'),horizons=('horizon_step','nunique'))
    wf=met[met.validation.str.startswith('walk_forward_')].groupby('model',as_index=False).agg(mae_w=('mae_w','mean'),rmse_w=('rmse_w','mean'),horizons=('horizon_step','nunique')); wf['validation']='walk_forward'; summary=pd.concat([summary,wf],ignore_index=True); summary.to_csv(OUT/'metrics_summary.csv',index=False)
    # charts
    for metric,label in [('mae_w','MAE'),('rmse_w','RMSE')]:
        pv=summary.pivot(index='model',columns='validation',values=metric); ax=pv.plot(kind='bar',figsize=(10,6)); ax.set_title(f'36-hour {label} — Models and Validation'); ax.set_ylabel(f'{label} (W)'); ax.set_xlabel('Model'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(OUT/f'model_validation_{label.lower()}.png',dpi=180); plt.close()
        sub=met[met.validation=='90_10']; plt.figure(figsize=(11,6));
        for model_name,g in sub.groupby('model'): plt.plot(g.horizon_hours,g[metric],label=model_name)
        plt.title(f'{label} Across 36-hour Forecast Horizon — 90/10'); plt.xlabel('Forecast horizon (hours)'); plt.ylabel(f'{label} (W)'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/f'horizon_{label.lower()}_90_10.png',dpi=180); plt.close()
    v=preds[(preds.validation=='90_10')&(preds.horizon_step==144)].dropna().tail(7*96)
    if len(v):
        plt.figure(figsize=(13,6));
        for col,label in [('actual_w','Actual'),('persistence_w','Persistence'),('copy_yesterday_w','Copy-Yesterday'),('ridge_w','Ridge'),('ridge_weather_w','Ridge + Weather')]: plt.plot(v.target_timestamp,v[col],label=label,alpha=.85)
        plt.title('Actual vs Forecast — 36-hour Horizon (90/10)'); plt.ylabel('Consumption (W)'); plt.legend(); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(OUT/'actual_vs_forecast_36h.png',dpi=180); plt.close()
        run = {
        'experiment': 'Prototype 2 - Ridge 36-Hour Forecasting Experiment',
        'experiment_stage': 'Extended experiment after Envitron Week 4 feedback on 29 September 2026',
        'building_id': BID,
        'description': desc,
        'forecast_horizon': '36 hours',
        'resolution': '15 minutes',
        'steps': 144,
        'validations': [
            '80/20 chronological',
            '90/10 chronological',
            '3-fold expanding walk-forward'
        ],
        'models': [
            'Persistence',
            'Copy-Yesterday',
            'Ridge',
            'Ridge + Weather'
        ],
        'metrics': ['MAE', 'RMSE'],
        'negative_handling': f'{len(neg)} negative rows excluded from modelling and exported separately; raw data unchanged.',
        'weather_note': 'The supplied dataset has historical observed weather, not archived future-weather forecasts. To prevent leakage, Ridge + Weather uses weather available at the forecast origin only.',
        'copy_yesterday_note': 'For horizons >24h, the baseline uses the most recent fully known same-slot daily profile (48h back), avoiding future leakage.',
        'runtime_seconds': time.perf_counter() - t
    }

    (OUT / 'run_summary.json').write_text(
        json.dumps(run, indent=2, default=str)
    )

    print(summary.to_string(index=False))
    print('\nRuntime', run['runtime_seconds'])


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()