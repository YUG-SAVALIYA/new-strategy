import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import time
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="Nifty 500 Strategy Backtester", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    .metric-card {
        background: rgba(22, 27, 34, 0.8); border: 1px solid #30363d; border-radius: 12px;
        padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title { font-size: 14px; color: #8b949e; text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
    .metric-value { font-size: 32px; font-weight: 800; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data(universe="Nifty 500"):
    nifty500_file = r'd:\New Strategy\ind_nifty500list.csv'
    try:
        n500_df = pd.read_csv(nifty500_file)
        valid_symbols = set(n500_df['Symbol'].str.strip().tolist())
    except Exception:
        valid_symbols = set()

    DATA_DIR = r'D:\New Strategy\groww_data'
    files = glob.glob(os.path.join(DATA_DIR, '*_Day.parquet'))
    data_dict = {}

    for f in files:
        sym = os.path.basename(f).replace('_Day.parquet', '')
        if universe == "Nifty 500" and sym not in valid_symbols: continue
        try:
            df = pd.read_parquet(f)
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi14'] = 100 - (100 / (1 + rs))
            
            df['sma50'] = df['close'].rolling(50).mean()
            
            sma20 = df['close'].rolling(20).mean()
            df['sma20'] = sma20
            std20 = df['close'].rolling(20).std()
            df['bb_width'] = (((sma20 + (2*std20)) - (sma20 - (2*std20))) / sma20) * 100
            df['dist_lower_bb'] = ((df['close'] - (sma20 - (2*std20))) / (sma20 - (2*std20))) * 100
            df['dist_sma50'] = ((df['close'] - df['sma50']) / df['sma50']) * 100
            
            df['vol_5d_avg'] = df['volume'].rolling(5).mean()
            df['vol_10d_avg'] = df['volume'].rolling(10).mean()
            
            df['total_range'] = df['high'] - df['low']
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['lower_wick_pct'] = np.where(df['total_range'] > 0, (df['lower_wick'] / df['total_range']) * 100, 0)
            
            data_dict[sym] = df
        except Exception:
            continue
            
    nifty = pd.read_csv(r'D:\New Strategy\nifty50_daily_candles.csv')
    nifty['date'] = pd.to_datetime(nifty['date']).dt.tz_localize(None)
    nifty = nifty.sort_values('date').reset_index(drop=True)
    nifty['sma50'] = nifty['close'].rolling(50).mean()
    macro_filters = {}
    for idx, row in nifty.iterrows():
        macro_filters[row['date'].date()] = row['close'] > row['sma50']
            
    return data_dict, macro_filters

@st.cache_data
def run_backtest(data_dict, macro_filters, use_macro, tm_mode, start_date, end_date):
    all_dates = pd.Series(pd.concat([df['datetime'] for df in data_dict.values()]).unique()).sort_values().reset_index(drop=True)
    calendar = all_dates[(all_dates.dt.date >= start_date) & (all_dates.dt.date <= end_date)].reset_index(drop=True)
    calendar_list = calendar.tolist()

    sig = {d: [] for d in calendar_list}

    for sym, df in data_dict.items():
        df_sub = df[(df['datetime'] >= pd.to_datetime(start_date) - pd.Timedelta(days=250)) & (df['datetime'] <= pd.to_datetime(end_date))]
        if df_sub.empty: continue
        
        cond = (df_sub['close'] > df_sub['sma50']) & \
               (df_sub['close'] > df_sub['sma20']) & \
               (df_sub['bb_width'] < 10.0) & \
               (df_sub['dist_sma50'] < 4.0) & \
               (df_sub['rsi14'] >= 45.0) & (df_sub['rsi14'] <= 65.0)
                    
        for d in df_sub[cond]['datetime']: 
            if use_macro:
                if d.date() not in macro_filters or not macro_filters[d.date()]: continue
            if d in sig: sig[d].append(sym)

    p_lookup = {sym: df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))].set_index('datetime')[['open', 'high', 'low', 'close']].to_dict('index') for sym, df in data_dict.items()}

    hd = 45
    sl = -5.0
    
    if tm_mode == "Fast Breathing (50%@8%, 50%@16%)":
        stg = [{'t': 8.0, 's': 50.0, 'nsl': -5.0}, {'t': 16.0, 's': 100.0, 'nsl': 0.0}]
    elif tm_mode == "All-or-Nothing 1:3 (100% @ 15%)":
        stg = [{'t': 15.0, 's': 100.0, 'nsl': 0.0}]
    elif tm_mode == "The Golden Hybrid (33%@10%, 67%@30%)":
        stg = [{'t': 10.0, 's': 33.0, 'nsl': -5.0}, {'t': 30.0, 's': 100.0, 'nsl': 0.0}]
    elif tm_mode == "Proper TM (50%@10%, Stay -5%)":
        stg = [{'t': 10.0, 's': 50.0, 'nsl': -5.0}, {'t': 20.0, 's': 100.0, 'nsl': 0.0}]
    else:
        stg = [{'t': 10.0, 's': 50.0, 'nsl': -5.0}, {'t': 20.0, 's': 100.0, 'nsl': 0.0}]
    
    equity = 100.0
    open_positions = []
    daily_ledger = []
    closed_trades = []
    
    alloc_frac = 0.20
    max_pos = 5
    max_pd = 5
    
    for i, current_date in enumerate(calendar_list):
        sod_deployed = sum(p['qty'] * p['close'] for p in open_positions) if open_positions else 0
        sod_cash = equity - sod_deployed
        
        if i > 0:
            taken = 0
            for sym in sorted(sig.get(calendar_list[i-1], [])):
                if taken >= max_pd or len(open_positions) >= max_pos: break
                if any(p['sym'] == sym for p in open_positions): continue
                if sym in p_lookup and current_date in p_lookup[sym]:
                    O = p_lookup[sym][current_date]['open']
                    
                    inv = equity * alloc_frac
                    if sod_cash >= inv:
                        qty = inv / O 
                        sod_cash -= inv
                        pend = [{'t': O*(1+s['t']/100.0), 's': s['s'], 'nsl': O*(1+s['nsl']/100.0) if s['nsl']!=0 else 0} for s in stg]
                        open_positions.append({
                            'sym': sym, 'ep': O, 'qty': qty, 'orig_qty': qty, 'days': 0, 
                            'close': p_lookup[sym][current_date]['close'], 
                            'csl': O * (1+sl/100.0), 'pend': pend
                        })
                        taken += 1
                        sod_deployed += inv
                        
        for p in open_positions: p['days'] += 1
        
        active = []
        for p in open_positions:
            sym = p['sym']
            if current_date not in p_lookup.get(sym, {}): active.append(p); continue
            day = p_lookup[sym][current_date]
            O, H, L, C = day['open'], day['high'], day['low'], day['close']
            qty, csl, pend = p['qty'], p['csl'], p['pend']
            
            def ex(ex_q, price, reason):
                nonlocal sod_cash, qty
                sod_cash += (ex_q * price)
                realized_pct = ((price - p['ep']) / p['ep']) * 100
                closed_trades.append({
                    'Symbol': sym, 
                    'Entry Date': calendar_list[i-p['days']], 
                    'Exit Date': current_date,
                    'Entry Price': p['ep'],
                    'Exit Price': price,
                    'Hold Days': p['days'],
                    'Reason': reason,
                    'Return %': realized_pct,
                    'Weight': ex_q / p['orig_qty']
                })
                qty -= ex_q
                
            if O <= csl: ex(qty, O, "SL (Open)")
            else:
                for s in [s for s in pend if O >= s['t']]:
                    sq = qty if s['s']>=99.99 else qty * (s['s']/100.0)
                    if s['nsl'] > csl: csl = s['nsl']
                    if sq > 0: ex(sq, O, "Target 1 (Open)")
                    pend.remove(s)
            if qty > 0:
                if L <= csl: ex(qty, csl, "SL (Intra)")
                else:
                    for s in [s for s in pend if H >= s['t']]:
                        sq = qty if s['s']>=99.99 else qty * (s['s']/100.0)
                        if s['nsl'] > csl: csl = s['nsl']
                        if sq > 0: ex(sq, s['t'], "Target (Intra)")
                        pend.remove(s)
            if qty > 0 and p['days'] >= hd: ex(qty, C, "Time Stop")
            
            p['qty'], p['csl'] = qty, csl
            if qty > 0:
                p['close'] = C
                active.append(p)
                
        open_positions = active
        equity = sod_cash + sum(p['qty']*p['close'] for p in open_positions)
        daily_ledger.append({'Date': current_date, 'Equity': equity})
        
    df = pd.DataFrame(daily_ledger)
    df['Running Peak'] = df['Equity'].cummax()
    df['Drawdown %'] = ((df['Equity'] / df['Running Peak']) - 1) * 100
    max_dd = abs(df['Drawdown %'].min())
    
    df_trades = pd.DataFrame(closed_trades)
    
    return df, df_trades, max_dd

st.title("⚡ Nifty 500 Blueprint Backtester")
st.markdown("Visualizing the performance using **20% allocation (Max 5 positions)**. Configure your Trade Management below.")

col_f1, col_f2 = st.columns(2)
with col_f1:
    use_macro = st.checkbox("🛡️ Enable Nifty 50 Macro Filter (> 50 SMA)", value=True)
    universe_mode = st.selectbox("🌌 Stock Universe", ["Nifty 500", "All Stocks (Full Database)"])
with col_f2:
    tm_mode = st.selectbox("⚙️ Trade Management Mode", [
        "Fast Breathing (50%@8%, 50%@16%)", 
        "All-or-Nothing 1:3 (100% @ 15%)",
        "The Golden Hybrid (33%@10%, 67%@30%)",
        "Proper TM (50%@10%, Stay -5%)", 
        "The 1/3rds Scaler (8%, 15%, 25%)"
    ])

col1, col2 = st.columns(2)
with col1:
    ui_start = st.date_input("Start Date", value=pd.to_datetime('2021-01-01').date(), min_value=pd.to_datetime('2010-01-01').date(), max_value=pd.to_datetime('2030-01-01').date())
with col2:
    ui_end = st.date_input("End Date", value=pd.to_datetime('2026-01-01').date(), min_value=pd.to_datetime('2010-01-01').date(), max_value=pd.to_datetime('2030-01-01').date())

if st.button("Run Portfolio Simulation", type="primary"):
    with st.spinner(f"Loading {universe_mode} Data (this may take a minute)..."):
        data_dict, macro_filters = load_and_prep_data(universe_mode)
    
    with st.spinner("Running Daily Portfolio Ledger..."):
        df_ledger, df_trades, max_dd = run_backtest(data_dict, macro_filters, use_macro, tm_mode, ui_start, ui_end)
        
    # Aggregate scale-outs into single blended trades for accurate Win Rate and Avg Return
    if len(df_trades) > 0:
        df_trades['Weighted Return'] = df_trades['Return %'] * df_trades['Weight']
        df_agg = df_trades.groupby(['Symbol', 'Entry Date']).agg(
            Weighted_Return=('Weighted Return', 'sum'),
            Hold_Days=('Hold Days', 'max'),
            Max_Return=('Return %', 'max')
        ).reset_index()
        df_agg.rename(columns={'Weighted_Return': 'Return %', 'Hold_Days': 'Hold Days', 'Max_Return': 'Max Return'}, inplace=True)
        
        win_rate = (len(df_agg[df_agg['Return %'] > 0]) / len(df_agg)) * 100
        aw = df_agg[df_agg['Return %'] > 0]['Return %'].mean() if len(df_agg[df_agg['Return %'] > 0]) > 0 else 0
        al = df_agg[df_agg['Return %'] <= 0]['Return %'].mean() if len(df_agg[df_agg['Return %'] <= 0]) > 0 else 0
    else:
        win_rate = aw = al = 0
        
    final_eq = df_ledger.iloc[-1]['Equity']
    
    # Calculate Full Target Hit Rate
    if tm_mode == "Fast Breathing (50%@8%, 50%@16%)": max_t = 15.9
    elif tm_mode == "All-or-Nothing 1:3 (100% @ 15%)": max_t = 14.9
    elif tm_mode == "The Golden Hybrid (33%@10%, 67%@30%)": max_t = 29.9
    elif tm_mode == "Proper TM (50%@10%, Stay -5%)": max_t = 19.9
    elif tm_mode == "The 1/3rds Scaler (8%, 15%, 25%)": max_t = 24.9
    else: max_t = 19.9
    
    total_trades = len(df_agg)
    full_target_hits = len(df_agg[df_agg['Max Return'] >= max_t])
    full_target_rate = (full_target_hits / total_trades) * 100 if total_trades > 0 else 0
    
    avg_ret_all = df_agg['Return %'].mean() if len(df_agg) > 0 else 0
    
    # 1. KPIs
    html = f'''
    <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; font-family: Arial, sans-serif;">
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Final Capital (Start 100)</div>
            <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">{final_eq:,.2f}</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Total Return</div>
            <div style="font-size: 24px; color: #3fb950; font-weight: bold;">+{(final_eq - 100):.1f}%</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Max Drawdown</div>
            <div style="font-size: 24px; color: #ff7b72; font-weight: bold;">{max_dd:.2f}%</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Win Rate</div>
            <div style="font-size: 24px; color: #58a6ff; font-weight: bold;">{win_rate:.1f}%</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Avg Win / Loss</div>
            <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">+{aw:.1f}% / {al:.1f}%</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Total Trades</div>
            <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">{len(df_agg)}</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Avg Return (Stock)</div>
            <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">+{avg_ret_all:.2f}%</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Avg Portfolio Growth</div>
            <div style="font-size: 24px; color: #3fb950; font-weight: bold;">+{(avg_ret_all * 0.20):.2f}%</div>
        </div>
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;" title="Of all trades taken, this % reached the final target">
            <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Hit Full Target</div>
            <div style="font-size: 24px; color: #d2a8ff; font-weight: bold;">{full_target_rate:.1f}%</div>
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

    # 2. Plotly Charts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ledger['Date'], y=df_ledger['Equity'], mode='lines', name='Normalized Equity', line=dict(color='#3fb950', width=2)))
    fig.add_trace(go.Scatter(x=df_ledger['Date'], y=df_ledger['Running Peak'], mode='lines', name='High Water Mark', line=dict(color='#8b949e', width=1, dash='dash')))
    fig.update_layout(title="Normalized Equity Curve", template='plotly_dark', height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    fig2 = px.area(df_ledger, x='Date', y='Drawdown %', title='Portfolio Drawdown (%)', color_discrete_sequence=['#ff7b72'])
    fig2.update_layout(template='plotly_dark', height=250, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, width='stretch')
    
    # 3. Yearly Breakdown
    st.subheader("Yearly Breakdown")
    df_ledger['Year'] = df_ledger['Date'].dt.year
    yearly = df_ledger.groupby('Year').agg(
        Start_Equity=('Equity', 'first'),
        End_Equity=('Equity', 'last')
    )
    yearly['Return %'] = ((yearly['End_Equity'] / yearly['Start_Equity']) - 1) * 100
    
    def get_max_dd(group):
        hwm = group['Equity'].cummax()
        dd = ((group['Equity'] / hwm) - 1) * 100
        return abs(dd.min())
        
    yearly['Max DD %'] = df_ledger.groupby('Year').apply(get_max_dd)
    
    # Add trade stats per year
    df_trades['Year'] = df_trades['Exit Date'].dt.year
    trade_stats = df_trades.groupby('Year').agg(
        Trades=('Symbol', 'count'),
        Wins=('Return %', lambda x: (x>0).sum()),
    )
    trade_stats['Win Rate %'] = (trade_stats['Wins'] / trade_stats['Trades']) * 100
    
    yearly = yearly.join(trade_stats).fillna(0)
    st.dataframe(yearly.style.format({
        'Start_Equity': '{:.2f}', 'End_Equity': '{:.2f}', 'Return %': '{:.2f}%', 
        'Max DD %': '{:.2f}%', 'Win Rate %': '{:.1f}%'
    }))

    # 4. Detailed Trade Log
    st.subheader("Detailed Trade Log")
    df_trades['Entry Date'] = df_trades['Entry Date'].dt.strftime('%Y-%m-%d')
    df_trades['Exit Date'] = df_trades['Exit Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(
        df_trades[['Symbol', 'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 'Hold Days', 'Reason', 'Return %']].style
        .format({'Entry Price': '₹{:.2f}', 'Exit Price': '₹{:.2f}', 'Return %': '{:.2f}%'})
        .map(lambda x: 'color: #3fb950' if isinstance(x, (int, float)) and x > 0 else ('color: #ff7b72' if isinstance(x, (int, float)) and x < 0 else ''), subset=['Return %'])
    )
