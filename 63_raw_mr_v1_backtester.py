import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import uuid
import json
import pickle
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="RAW MR V1 Research Platform")

st.title("RAW MR V1 Research Platform")

# --- Initialize Session State ---
default_stages_df = pd.DataFrame({
    'Enable': [True, True, True],
    'Trigger %': [5.0, 10.0, 20.0],
    'Sell Qty %': [0.0, 25.0, 50.0],
    'New SL %': [0.0, 3.0, 10.0]
})

SAVE_DIR = "saved_runs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if 'saved_runs' not in st.session_state:
    st.session_state.saved_runs = {}
    for f in glob.glob(os.path.join(SAVE_DIR, "*.pkl")):
        try:
            with open(f, 'rb') as file:
                run_data = pickle.load(file)
                run_name = run_data.get('run_name', os.path.basename(f).replace('.pkl', ''))
                st.session_state.saved_runs[run_name] = run_data
        except Exception:
            pass

if 'init_capital' not in st.session_state:
    st.session_state.update({
        'init_capital': 1000000,
        'start_date': pd.to_datetime('2015-01-01').date(),
        'end_date': pd.to_datetime('2026-01-01').date(),
        'sma_period': 20,
        'mr_threshold_pct': 25.0,
        'hold_period': 20,
        'alloc_pct': 10.0,
        'max_positions': 10,
        'fees_pct': 0.05,
        'slippage_pct': 0.05,
        'use_initial_sl': False,
        'initial_sl_pct': -5.0,
        'editor_key_counter': 0
    })

def load_frozen_preset():
    st.session_state.update({
        'sma_period': 20,
        'mr_threshold_pct': 25.0,
        'hold_period': 20,
        'use_initial_sl': False,
        'editor_key_counter': st.session_state.get('editor_key_counter', 0) + 1
    })

# --- UI Parameters ---
st.sidebar.button("Load Frozen RAW MR V1 Preset", on_click=load_frozen_preset, help="Restores 20 SMA, 25% drop, 20-day hold, no SL, no stages.")

st.sidebar.header("Strategy Parameters")
init_capital = st.sidebar.number_input("Initial Capital", key='init_capital')
start_date = st.sidebar.date_input("Start Date", key='start_date')
end_date = st.sidebar.date_input("End Date", key='end_date')
sma_period = st.sidebar.number_input("SMA Period", key='sma_period', step=1)
mr_threshold_pct = st.sidebar.number_input("Mean-Reversion Threshold %", key='mr_threshold_pct')
hold_period = st.sidebar.number_input("Holding Period (days)", key='hold_period', step=1)

st.sidebar.header("Portfolio Parameters")
alloc_pct = st.sidebar.number_input("Allocation % of Available Cash", key='alloc_pct')
max_positions = st.sidebar.number_input("Maximum Concurrent Positions", key='max_positions', step=1)
fees_pct = st.sidebar.number_input("Fees %", key='fees_pct')
slippage_pct = st.sidebar.number_input("Slippage %", key='slippage_pct')

st.sidebar.header("Risk Management")
use_initial_sl = st.sidebar.checkbox("Use Initial Stop Loss", key='use_initial_sl')
initial_sl_pct = st.sidebar.number_input("Initial Stop Loss %", key='initial_sl_pct', disabled=not use_initial_sl)

st.sidebar.markdown("### Upside Stages")
# Use a dynamic key based on a counter to allow complete resetting of the data editor without triggering assignment errors
editor_key = f"stages_df_{st.session_state.get('editor_key_counter', 0)}"
edited_stages = st.sidebar.data_editor(default_stages_df, num_rows="dynamic", key=editor_key)
enabled_stages = edited_stages[edited_stages['Enable'] == True].sort_values('Trigger %').to_dict('records')

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Filters")
min_simul_signals = st.sidebar.number_input("Min Signals Per Day", min_value=1, value=1, help="Only take trades if at least this many stocks trigger on the same day.")
skip_friday_entries = st.sidebar.checkbox("Skip Friday Entries", value=False, help="Do not buy stocks on Fridays to avoid weekend gap risk.")
min_market_cap = st.sidebar.number_input("Min Market Cap (₹ Crores)", min_value=0, value=8000, help="Ignore stocks below this market cap.")

mr_mult = 1.0 - (mr_threshold_pct / 100.0)
alloc_frac = alloc_pct / 100.0
fees_frac = fees_pct / 100.0
slippage_frac = slippage_pct / 100.0

@st.cache_data
def load_data(data_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_dir = os.path.join(script_dir, data_dir)
    files = glob.glob(os.path.join(full_data_dir, "*_Day.parquet"))
    dfs = {}
    for f in files:
        symbol = os.path.basename(f).replace("_Day.parquet", "")
        df = pd.read_parquet(f)
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
        dfs[symbol] = df.sort_values('datetime').reset_index(drop=True)
    return dfs

@st.cache_data
def load_market_cap():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Average Market cap.xlsx")
    if not os.path.exists(file_path):
        return {}
    df = pd.read_excel(file_path)
    col = [c for c in df.columns if 'Average market capitalisation' in c]
    if not col: return {}
    col = col[0]
    df = df.dropna(subset=[col, 'Symbol'])
    return {str(row['Symbol']).strip(): float(row[col]) / 100.0 for _, row in df.iterrows()}

with st.spinner("Loading Data..."):
    data_dict = load_data('groww_data')
    mcap_dict = load_market_cap()

if not data_dict:
    st.error("⚠️ No data files found! Please ensure your `.parquet` data files are placed in the `groww_data/` folder.")
    st.stop()

# Build trading calendar
all_dates = pd.Series(pd.concat([df['datetime'] for df in data_dict.values()]).unique()).sort_values().reset_index(drop=True)
calendar = all_dates[(all_dates.dt.date >= start_date) & (all_dates.dt.date <= end_date)].reset_index(drop=True)
calendar_list = calendar.tolist()

if len(calendar_list) == 0:
    st.error("No trading days found in the selected date range.")
    st.stop()

# --- Reconciliation Function ---
def verify_reconciliation(df_ledger, df_trades, open_positions):
    errors = []
    
    # 1. Cash + market value = total equity
    diff1 = np.abs(df_ledger['Available Cash'] + df_ledger['Deployed Capital'] + df_ledger['Unrealized P&L'] - df_ledger['Total Equity'])
    if (diff1 > 0.01).any():
        errors.append("Cash + Market Value != Total Equity")
        
    # 2. Daily P&L = equity change
    equity_diff = df_ledger['Total Equity'].diff()
    equity_diff.iloc[0] = df_ledger['Total Equity'].iloc[0] - init_capital
    diff2 = np.abs(df_ledger['Net Daily P&L'] - equity_diff)
    if (diff2 > 0.01).any():
        errors.append("Daily P&L != Equity Change")
        
    # 3. Monthly totals = daily totals
    df_ledger['Month_Str'] = df_ledger['Date'].dt.to_period('M')
    
    # 4. Yearly totals = monthly totals
    df_ledger['Year_Str'] = df_ledger['Date'].dt.to_period('Y')
    
    # 5. Fees/slippage counted once
    total_ledger_fees = df_ledger['Fees'].sum()
    total_trade_fees = df_trades['Fees'].sum() if not df_trades.empty else 0
    remaining_open_fees = sum(p['total_entry_fees'] * (p['current_qty'] / p['entry_qty']) for p in open_positions) if open_positions else 0
    if abs(total_ledger_fees - (total_trade_fees + remaining_open_fees)) > 0.01:
        errors.append(f"Fees counted incorrectly. Ledger: {total_ledger_fees}, Trades+Open: {total_trade_fees + remaining_open_fees}")
        
    total_ledger_slip = df_ledger['Slippage'].sum()
    total_trade_slip = df_trades['Slippage'].sum() if not df_trades.empty else 0
    remaining_open_slip = sum(p['total_entry_slippage'] * (p['current_qty'] / p['entry_qty']) for p in open_positions) if open_positions else 0
    if abs(total_ledger_slip - (total_trade_slip + remaining_open_slip)) > 0.01:
        errors.append(f"Slippage counted incorrectly. Ledger: {total_ledger_slip}, Trades+Open: {total_trade_slip + remaining_open_slip}")
        
    # 6. No negative shares & Partial exits <= remaining quantity
    if (df_ledger['Open Positions'] < 0).any():
        errors.append("Negative open positions in ledger")
    if not df_trades.empty:
        if (df_trades['qty'] < 0).any() or (df_trades['Remaining Qty'] < 0).any():
            errors.append("Negative shares or partial exits exceeded remaining quantity")
            
    # 7. No duplicate stage execution
    if not df_trades.empty:
        counts = df_trades.groupby(['trade_id', 'exit_reason']).size()
        if (counts > 1).any():
            errors.append("Duplicate stage execution detected for a single trade")
            
    # 8. No exit before entry
    if not df_trades.empty:
        if (df_trades['exit_date'] < df_trades['entry_date']).any():
            errors.append("Exit date before entry date detected")
            
    # 9. No future data used
    if not df_trades.empty:
        if (df_trades['entry_date'] <= df_trades['signal_date']).any():
            errors.append("Entry happens on or before signal date (Lookahead bias)")
            
    return errors

@st.cache_data
def precompute_signals(sma_p, mr_thresh, min_mcap, s_date, e_date, cal_list):
    mr_m = 1.0 - (mr_thresh / 100.0)
    sig_by_date = {d: [] for d in cal_list}
    lookback = int(sma_p) * 2 + 50
    for sym, df in data_dict.items():
        if mcap_dict.get(sym, 0) < min_mcap: continue
        
        df_sub = df[(df['datetime'] >= pd.to_datetime(s_date) - pd.Timedelta(days=lookback)) & (df['datetime'] <= pd.to_datetime(e_date))]
        if df_sub.empty: continue
        
        df_sub = df_sub.copy()
        df_sub['sma'] = df_sub['close'].rolling(int(sma_p)).mean()
        df_sub['thresh'] = df_sub['sma'] * mr_m
        
        df_sub['close_prev'] = df_sub['close'].shift(1)
        df_sub['thresh_prev'] = df_sub['thresh'].shift(1)
        
        cond1 = df_sub['close_prev'] <= df_sub['thresh_prev']
        cond2 = df_sub['close'] > df_sub['close_prev']
        
        df_sub['signal'] = cond1 & cond2
        
        sig_dates = df_sub[df_sub['signal']]['datetime'].tolist()
        for d in sig_dates:
            if d in sig_by_date:
                sig_by_date[d].append(sym)
    return sig_by_date

@st.cache_data
def precompute_price_lookup(s_date, e_date):
    p_lookup = {}
    for sym, df in data_dict.items():
        df_sub = df[(df['datetime'] >= pd.to_datetime(s_date)) & (df['datetime'] <= pd.to_datetime(e_date))].copy()
        df_sub['date_val'] = df_sub['datetime']
        df_sub.set_index('date_val', inplace=True)
        p_lookup[sym] = df_sub[['open', 'high', 'low', 'close']].to_dict('index')
    return p_lookup

def run_simulation():
    signals_by_date = precompute_signals(sma_period, mr_threshold_pct, min_market_cap, start_date, end_date, calendar_list)
    price_lookup = precompute_price_lookup(start_date, end_date)

    cash = init_capital
    open_positions = []
    all_trades = []
    daily_ledger = []
    trade_id_counter = 1
    
    for i, current_date in enumerate(calendar_list):
        new_trades_today = 0
        buys_today = []
        buys_today_raw = []
        daily_entry_fees = 0
        daily_entry_slip = 0
        
        # Calculate Start of Day Equity for equal position sizing
        sod_deployed = sum(pos['current_qty'] * pos['current_close'] for pos in open_positions) if open_positions else 0
        sod_equity = cash + sod_deployed
        
        # 1. Evaluate entries from D-1 signals
        if i > 0:
            prev_date = calendar_list[i-1]
            signals = signals_by_date.get(prev_date, [])
            
            if len(signals) < min_simul_signals:
                signals = []
                
            if skip_friday_entries and current_date.weekday() == 4:
                signals = []
                
            signals = sorted(signals)
            
            for sym in signals:
                if len(open_positions) >= int(max_positions): break
                    
                if sym in price_lookup and current_date in price_lookup[sym]:
                    today_prices = price_lookup[sym][current_date]
                    entry_price = today_prices['open']
                    
                    ideal_alloc = sod_equity * alloc_frac
                    alloc = min(ideal_alloc, cash)
                    qty = math.floor(alloc / entry_price)
                    
                    if qty > 0:
                        investment = qty * entry_price
                        fee = investment * fees_frac
                        slip = investment * slippage_frac
                        total_cost = investment + fee + slip
                        
                        if cash >= total_cost:
                            cash -= total_cost
                            daily_entry_fees += fee
                            daily_entry_slip += slip
                            buys_today.append(f"{sym}: {qty} @ ₹{entry_price:.2f}")
                            buys_today_raw.append({
                                'symbol': sym, 'qty': qty, 'price': entry_price,
                                'value': total_cost, 'fee': fee + slip
                            })
                            
                            new_trade = {
                                'trade_id': trade_id_counter,
                                'symbol': sym,
                                'signal_date': prev_date,
                                'entry_date': current_date,
                                'entry_price': entry_price,
                                'entry_qty': qty,
                                'current_qty': qty,
                                'total_entry_fees': fee,
                                'total_entry_slippage': slip,
                                'days_held': 0,
                                'mfe_price': today_prices['high'],
                                'mae_price': today_prices['low'],
                                'current_close': today_prices['close'],
                                'current_sl': entry_price * (1 + initial_sl_pct / 100.0) if use_initial_sl else None,
                                'pending_stages': []
                            }
                            
                            for stg_idx, stg_row in enumerate(enabled_stages):
                                new_trade['pending_stages'].append({
                                    'stage_num': stg_idx + 1,
                                    'trigger_price': entry_price * (1 + stg_row['Trigger %'] / 100.0),
                                    'sell_qty_pct': stg_row['Sell Qty %'],
                                    'new_sl_price': entry_price * (1 + stg_row['New SL %'] / 100.0) if stg_row['New SL %'] != 0 else entry_price
                                })
                                
                            open_positions.append(new_trade)
                            trade_id_counter += 1
                            new_trades_today += 1
                            
        # 2. Increment days_held
        for pos in open_positions:
            pos['days_held'] += 1
            
        # 3. Evaluate Exits
        exits_today = []
        active_after_exit = []
        
        daily_exit_fees = 0
        daily_exit_slip = 0
        daily_gross_pnl = 0
        
        for pos in open_positions:
            sym = pos['symbol']
            if current_date not in price_lookup.get(sym, {}):
                active_after_exit.append(pos)
                continue
                
            prices = price_lookup[sym][current_date]
            O, H, L, C = prices['open'], prices['high'], prices['low'], prices['close']
            
            pos['mfe_price'] = max(pos['mfe_price'], H)
            pos['mae_price'] = min(pos['mae_price'], L)
            
            qty_remaining = pos['current_qty']
            current_sl = pos['current_sl']
            pending_stages = pos['pending_stages']
            
            def execute_exit(exit_qty, exit_price, reason, new_sl_val, trigger_price_val=None):
                nonlocal cash, daily_exit_fees, daily_exit_slip, daily_gross_pnl, qty_remaining
                
                gross_pnl = (exit_price - pos['entry_price']) * exit_qty
                exit_val = exit_qty * exit_price
                
                frac = exit_qty / pos['entry_qty']
                prorated_entry_fee = pos['total_entry_fees'] * frac
                prorated_entry_slip = pos['total_entry_slippage'] * frac
                
                exit_fee = exit_val * fees_frac
                exit_slip = exit_val * slippage_frac
                
                net_pnl = gross_pnl - prorated_entry_fee - prorated_entry_slip - exit_fee - exit_slip
                
                cash += (exit_val - exit_fee - exit_slip)
                
                daily_exit_fees += exit_fee
                daily_exit_slip += exit_slip
                daily_gross_pnl += gross_pnl
                
                qty_remaining -= exit_qty
                
                tr = {
                    'trade_id': pos['trade_id'],
                    'symbol': pos['symbol'],
                    'signal_date': pos['signal_date'],
                    'entry_date': pos['entry_date'],
                    'entry_price': pos['entry_price'],
                    'qty': exit_qty,
                    'investment': exit_qty * pos['entry_price'],
                    'exit_date': current_date,
                    'Trigger Price': trigger_price_val,
                    'exit_price': exit_price,
                    'exit_reason': reason,
                    'days_held': pos['days_held'],
                    'gross_pnl': gross_pnl,
                    'entry_fees': prorated_entry_fee,
                    'entry_slippage': prorated_entry_slip,
                    'Fees': prorated_entry_fee + exit_fee,
                    'Slippage': prorated_entry_slip + exit_slip,
                    'net_pnl': net_pnl,
                    'return_pct': (net_pnl / (exit_qty * pos['entry_price'])) * 100,
                    'MFE %': (pos['mfe_price'] - pos['entry_price']) / pos['entry_price'] * 100,
                    'MAE %': (pos['mae_price'] - pos['entry_price']) / pos['entry_price'] * 100,
                    'New SL': new_sl_val,
                    'Remaining Qty': qty_remaining
                }
                all_trades.append(tr)
                exits_today.append(tr)

            # OPEN hit check
            open_hit_sl = (current_sl is not None) and (O <= current_sl)
            open_stages = [s for s in pending_stages if O >= s['trigger_price']] if not open_hit_sl else []
            
            if open_hit_sl:
                execute_exit(qty_remaining, O, "SL (Open)", current_sl, trigger_price_val=current_sl)
            elif open_stages:
                for stg in open_stages:
                    if qty_remaining <= 0: break
                    sell_pct = stg['sell_qty_pct']
                    if sell_pct == 0:
                        exit_qty = 0
                    else:
                        exit_qty = math.floor(qty_remaining * (sell_pct / 100.0))
                        if sell_pct >= 99.99: exit_qty = qty_remaining
                        if exit_qty == 0 and qty_remaining > 0: exit_qty = 1
                        exit_qty = min(exit_qty, qty_remaining)
                    
                    new_sl = stg['new_sl_price']
                    if current_sl is None or new_sl > current_sl:
                        current_sl = new_sl
                        pos['current_sl'] = current_sl
                        
                    if exit_qty > 0:
                        execute_exit(exit_qty, O, f"Stage {stg['stage_num']} (Open)", current_sl, trigger_price_val=stg['trigger_price'])
                        
                    pending_stages.remove(stg)
                    
            # INTRADAY hit check
            if qty_remaining > 0:
                intra_hit_sl = (current_sl is not None) and (L <= current_sl)
                intra_stages = [s for s in pending_stages if H >= s['trigger_price']] if not intra_hit_sl else []
                
                if intra_hit_sl:
                    execute_exit(qty_remaining, current_sl, "SL (Intra)", current_sl, trigger_price_val=current_sl)
                elif intra_stages:
                    for stg in intra_stages:
                        if qty_remaining <= 0: break
                        sell_pct = stg['sell_qty_pct']
                        if sell_pct == 0:
                            exit_qty = 0
                        else:
                            exit_qty = math.floor(qty_remaining * (sell_pct / 100.0))
                            if sell_pct >= 99.99: exit_qty = qty_remaining
                            if exit_qty == 0 and qty_remaining > 0: exit_qty = 1
                            exit_qty = min(exit_qty, qty_remaining)
                        
                        new_sl = stg['new_sl_price']
                        if current_sl is None or new_sl > current_sl:
                            current_sl = new_sl
                            pos['current_sl'] = current_sl
                            
                        if exit_qty > 0:
                            execute_exit(exit_qty, stg['trigger_price'], f"Stage {stg['stage_num']} (Intra)", current_sl, trigger_price_val=stg['trigger_price'])
                        
                        pending_stages.remove(stg)
                        
            # TIME STOP check
            if qty_remaining > 0 and pos['days_held'] >= int(hold_period):
                execute_exit(qty_remaining, C, "Time Stop", current_sl, trigger_price_val=None)
                
            # EOD position finalize
            pos['current_qty'] = qty_remaining
            if qty_remaining > 0:
                pos['current_close'] = C
                active_after_exit.append(pos)
                
        open_positions = active_after_exit
        
        # 3. EOD Accounting
        deployed_cap = 0
        unrealized_pnl = 0
        
        for pos in open_positions:
            current_val = pos['current_qty'] * pos['current_close']
            inv = pos['current_qty'] * pos['entry_price']
            deployed_cap += inv
            unrealized_pnl += (current_val - inv)
            
        total_equity = cash + deployed_cap + unrealized_pnl
        
        sells_today_str = [f"{tr['symbol']}: {tr['qty']} @ ₹{tr['exit_price']:.2f} ({tr['exit_reason']})" for tr in exits_today]
        sells_today_raw = [{
            'symbol': tr['symbol'], 'qty': tr['qty'], 'price': tr['exit_price'],
            'value': tr['qty'] * tr['exit_price'], 'reason': tr['exit_reason'], 'pnl': tr['net_pnl']
        } for tr in exits_today]
        
        holdings_str = [f"{pos['symbol']}: {pos['current_qty']} @ ₹{pos['current_close']:.2f} (Inv: ₹{pos['current_qty']*pos['entry_price']:.2f})" for pos in open_positions]
        holdings_raw = [{
            'symbol': pos['symbol'], 'qty': pos['current_qty'], 'close': pos['current_close'],
            'inv': pos['current_qty'] * pos['entry_price']
        } for pos in open_positions]
        
        wins_today = sum(1 for tr in exits_today if tr['net_pnl'] > 0)
        losses_today = sum(1 for tr in exits_today if tr['net_pnl'] <= 0)
        
        target_hits = sum(1 for tr in exits_today if 'Stage' in tr['exit_reason'])
        sl_hits = sum(1 for tr in exits_today if 'SL' in tr['exit_reason'])
        time_stops = sum(1 for tr in exits_today if 'Time Stop' in tr['exit_reason'])
        
        gross_profit_today = sum(tr['net_pnl'] for tr in exits_today if tr['net_pnl'] > 0)
        gross_loss_today = abs(sum(tr['net_pnl'] for tr in exits_today if tr['net_pnl'] < 0))
        pf_today = gross_profit_today / gross_loss_today if gross_loss_today > 0 else (float('inf') if gross_profit_today > 0 else 0)
        
        avg_ret_today = sum(tr['return_pct'] for tr in exits_today) / len(exits_today) if exits_today else 0
        
        daily_ledger.append({
            'Date': current_date,
            'Available Cash': cash,
            'Deployed Capital': deployed_cap,
            'Open Positions': len(open_positions),
            'New Trades': new_trades_today,
            'Exits': len(exits_today),
            'Realized P&L': daily_gross_pnl,
            'Unrealized P&L': unrealized_pnl,
            'Fees': daily_entry_fees + daily_exit_fees,
            'Slippage': daily_entry_slip + daily_exit_slip,
            'Total Equity': total_equity,
            'Buys': ", ".join(buys_today),
            'Buys_Raw': buys_today_raw,
            'Sells': ", ".join(sells_today_str),
            'Sells_Raw': sells_today_raw,
            'Holdings': ", ".join(holdings_str),
            'Holdings_Raw': holdings_raw,
            'Wins': wins_today,
            'Losses': losses_today,
            'Target Hits': target_hits,
            'SL Hits': sl_hits,
            'Time Stops': time_stops,
            'Avg Return %': avg_ret_today,
            'Profit Factor': pf_today
        })
        
    return daily_ledger, all_trades, open_positions

# --- Tabs ---
tab_backtest, tab_compare = st.tabs(["Backtester", "Run Comparison"])

with tab_backtest:
    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Run Backtest", type="primary")
    with col2:
        if st.session_state.saved_runs:
            load_run = st.selectbox("Load Saved Run Results", ["-- Select a saved run to view --"] + list(st.session_state.saved_runs.keys()))
        else:
            load_run = "-- Select a saved run to view --"

    if run_btn:
        with st.spinner("Running Simulation..."):
            daily_ledger, all_trades, open_positions = run_simulation()

            # --- Post-Processing ---
            df_ledger = pd.DataFrame(daily_ledger)
            df_ledger['Net Daily P&L'] = df_ledger['Total Equity'].diff()
            df_ledger.loc[0, 'Net Daily P&L'] = df_ledger.loc[0, 'Total Equity'] - init_capital
            
            df_ledger['Daily Return'] = df_ledger['Total Equity'].pct_change().fillna(0)
            df_ledger['Cumulative Return'] = (1 + df_ledger['Daily Return']).cumprod() - 1
            df_ledger['High Water Mark'] = df_ledger['Total Equity'].cummax()
            df_ledger['Drawdown ₹'] = df_ledger['High Water Mark'] - df_ledger['Total Equity']
            df_ledger['Drawdown %'] = (df_ledger['Drawdown ₹'] / df_ledger['High Water Mark']) * 100
            
            # Add Cumulative Metrics
            df_ledger['Cum Wins'] = df_ledger['Wins'].cumsum()
            df_ledger['Cum Losses'] = df_ledger['Losses'].cumsum()
            df_ledger['Cum Target Hits'] = df_ledger['Target Hits'].cumsum()
            df_ledger['Cum SL Hits'] = df_ledger['SL Hits'].cumsum()

            df_trades = pd.DataFrame(all_trades)
            if not df_trades.empty:
                df_trades['Holding Days'] = df_trades['days_held']
                df_trades = df_trades[['trade_id', 'symbol', 'signal_date', 'entry_date', 'entry_price', 'qty', 'investment', 
                                       'exit_date', 'Trigger Price', 'exit_price', 'exit_reason', 'Holding Days', 'gross_pnl', 'Fees', 'Slippage', 
                                       'net_pnl', 'return_pct', 'MFE %', 'MAE %', 'New SL', 'Remaining Qty']]
            
            # Monthly & Yearly Ledger
            df_ledger['Month'] = df_ledger['Date'].dt.to_period('M')
            df_ledger['Year'] = df_ledger['Date'].dt.to_period('Y')
            
            def agg_ledger(df, period_col):
                agg = df.groupby(period_col).agg(
                    Opening_Equity=('Total Equity', 'first'),
                    Closing_Equity=('Total Equity', 'last'),
                    Net_PnL=('Net Daily P&L', 'sum'),
                    Max_DD=('Drawdown %', 'max'),
                    Fees_Total=('Fees', 'sum'),
                    Slippage_Total=('Slippage', 'sum')
                ).reset_index()
                # Correct Opening Equity to match previous period's closing
                agg['Opening_Equity'] = agg['Closing_Equity'] - agg['Net_PnL']
                agg['Return %'] = (agg['Closing_Equity'] - agg['Opening_Equity']) / agg['Opening_Equity'] * 100
                
                if not df_trades.empty:
                    df_t = df_trades.copy()
                    df_t['Period'] = df_t['exit_date'].dt.to_period(period_col[0])
                    t_agg = df_t.groupby('Period').agg(
                        Trades=('trade_id', 'count'),
                        Wins=('net_pnl', lambda x: (x > 0).sum()),
                        Losses=('net_pnl', lambda x: (x <= 0).sum()),
                        Target_Hits=('exit_reason', lambda x: x.str.contains('Stage').sum()),
                        SL_Hits=('exit_reason', lambda x: x.str.contains('SL').sum()),
                        Time_Stops=('exit_reason', lambda x: (x == 'Time Stop').sum()),
                        Gross_Profit=('net_pnl', lambda x: x[x>0].sum()),
                        Gross_Loss=('net_pnl', lambda x: abs(x[x<=0].sum())),
                        Avg_Return_Pct=('return_pct', 'mean')
                    ).reset_index()
                    agg = agg.merge(t_agg, left_on=period_col, right_on='Period', how='left').drop(columns=['Period']).fillna(0)
                    agg['Win Rate'] = (agg['Wins'] / agg['Trades'] * 100).fillna(0)
                    agg['Profit Factor'] = np.where(agg['Gross_Loss'] == 0, 
                                                    np.where(agg['Gross_Profit'] == 0, 0, np.inf), 
                                                    agg['Gross_Profit'] / agg['Gross_Loss'])
                else:
                    for col in ['Trades', 'Wins', 'Losses', 'Target_Hits', 'SL_Hits', 'Time_Stops', 'Win Rate', 'Gross_Profit', 'Gross_Loss', 'Profit Factor', 'Avg_Return_Pct']:
                        agg[col] = 0
                return agg

            df_ledger['Week'] = df_ledger['Date'].dt.to_period('W')
            
            df_monthly = agg_ledger(df_ledger, 'Month')
            df_yearly = agg_ledger(df_ledger, 'Year')
            df_weekly = agg_ledger(df_ledger, 'Week')
            
            df_monthly['Month'] = df_monthly['Month'].astype(str)
            df_yearly['Year'] = df_yearly['Year'].astype(str)
            df_weekly['Week'] = df_weekly['Week'].astype(str)

            # --- Validation ---
            validation_errors = verify_reconciliation(df_ledger, df_trades, open_positions)

            # --- KPIs ---
            final_cap = df_ledger['Total Equity'].iloc[-1]
            total_ret = (final_cap / init_capital - 1) * 100
            days = (df_ledger['Date'].iloc[-1] - df_ledger['Date'].iloc[0]).days
            cagr = ((final_cap / init_capital) ** (365.25 / days) - 1) * 100 if days > 0 else 0
            
            total_trades = len(df_trades)
            wins = len(df_trades[df_trades['net_pnl'] > 0]) if total_trades > 0 else 0
            losses = total_trades - wins
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            
            target_hits_total = df_trades['exit_reason'].str.contains('Stage').sum() if total_trades > 0 else 0
            sl_hits_total = df_trades['exit_reason'].str.contains('SL').sum() if total_trades > 0 else 0
            time_stops_total = (df_trades['exit_reason'] == 'Time Stop').sum() if total_trades > 0 else 0
            
            gross_profit = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum() if total_trades > 0 else 0
            gross_loss = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum()) if total_trades > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
            
            avg_trade = df_trades['net_pnl'].mean() if total_trades > 0 else 0
            avg_ret_pct = df_trades['return_pct'].mean() if total_trades > 0 else 0
            
            max_dd_pct = df_ledger['Drawdown %'].max()
            max_dd_rs = df_ledger['Drawdown ₹'].max()
            
            # --- Store Current Run in Session State ---
            kpis = {
                'Final Capital': final_cap,
                'Total Return %': total_ret,
                'CAGR %': cagr,
                'Max DD %': max_dd_pct,
                'Max DD ₹': max_dd_rs,
                'Total Trades': total_trades,
                'Win Rate %': win_rate,
                'Wins': wins,
                'Losses': losses,
                'Target Hits': target_hits_total,
                'SL Hits': sl_hits_total,
                'Time Stops': time_stops_total,
                'Profit Factor': profit_factor,
                'Avg Trade P&L': avg_trade,
                'Avg Trade Ret %': avg_ret_pct
            }
            
            st.session_state.current_results = {
                'df_ledger': df_ledger,
                'df_trades': df_trades,
                'df_monthly': df_monthly,
                'df_yearly': df_yearly,
                'df_weekly': df_weekly,
                'kpis': kpis,
                'validation_errors': validation_errors,
                'params': {
                    'init_capital': init_capital, 'start_date': str(start_date), 'end_date': str(end_date),
                    'sma_period': sma_period, 'mr_threshold_pct': mr_threshold_pct, 'hold_period': hold_period,
                    'alloc_pct': alloc_pct, 'max_positions': max_positions, 'fees_pct': fees_pct, 'slippage_pct': slippage_pct,
                    'use_initial_sl': use_initial_sl, 'initial_sl_pct': initial_sl_pct,
                    'enabled_stages': enabled_stages, 'min_simul_signals': min_simul_signals, 'skip_friday': skip_friday_entries,
                    'min_market_cap': min_market_cap
                }
            }

    elif load_run != "-- Select a saved run to view --":
        saved = st.session_state.saved_runs[load_run]
        if 'results' in saved:
            st.session_state.current_results = saved['results']
        else:
            st.warning("⚠️ This run was saved before the detailed results feature was added. Please re-run it to view details.")

    if 'current_results' in st.session_state:
        res = st.session_state.current_results
        df_ledger = res['df_ledger']
        df_trades = res['df_trades']
        df_monthly = res['df_monthly']
        df_yearly = res['df_yearly']
        df_weekly = res.get('df_weekly', pd.DataFrame())
        kpis = res['kpis']
        validation_errors = res['validation_errors']

        if validation_errors:
            st.error("### 🚨 BACKTEST FAILED RECONCILIATION")
            for err in validation_errors:
                st.error(f"- {err}")
            st.stop()
        else:
            st.success("✅ Run verified! All ledgers perfectly reconcile.")

        # --- Display ---
        def render_kpis(kpis):
            html = f'''
            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; font-family: Arial, sans-serif;">
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Final Capital</div>
                    <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">₹{kpis['Final Capital']:,.2f}</div>
                    <div style="font-size: 12px; color: #3fb950; margin-top: 5px;">Return: {kpis['Total Return %']:.2f}%</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">CAGR</div>
                    <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">{kpis['CAGR %']:.2f}%</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Max Drawdown</div>
                    <div style="font-size: 24px; color: #ff7b72; font-weight: bold;">{kpis['Max DD %']:.2f}%</div>
                    <div style="font-size: 12px; color: #8b949e; margin-top: 5px;">₹-{kpis['Max DD ₹']:,.2f}</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Win Rate</div>
                    <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">{kpis['Win Rate %']:.2f}%</div>
                    <div style="font-size: 12px; color: #8b949e; margin-top: 5px;">{kpis['Total Trades']} Trades</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Win / Loss</div>
                    <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;"><span style="color:#3fb950">{kpis['Wins']}</span> / <span style="color:#ff7b72">{kpis['Losses']}</span></div>
                    <div style="font-size: 12px; color: #8b949e; margin-top: 5px;">Avg Ret: {kpis['Avg Trade Ret %']:.2f}% (₹{kpis['Avg Trade P&L']:,.2f})</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Exits Breakdown</div>
                    <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">{kpis['Target Hits']} / {kpis['SL Hits']}</div>
                    <div style="font-size: 12px; color: #8b949e; margin-top: 5px;">Target / SL Hits ({kpis['Time Stops']} Time Stops)</div>
                </div>
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; flex: 1; min-width: 150px;">
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: bold; margin-bottom: 5px;">Profit Factor</div>
                    <div style="font-size: 24px; color: #c9d1d9; font-weight: bold;">{kpis['Profit Factor']:.2f}</div>
                </div>
            </div>
            '''
            return html

        def render_custom_ledger(df):
            # Reverse order to show newest first
            df_slice = df.iloc[::-1]
            
            html = ['<div style="height: 800px; overflow-y: auto; background-color: #0e1117; padding: 10px; font-family: Arial, sans-serif;">']
            html.append('<table style="width: 100%; border-collapse: collapse; color: #c9d1d9;">')
            html.append('''
                <thead>
                    <tr style="border-bottom: 1px solid #30363d; text-align: left; color: #8b949e; font-size: 12px;">
                        <th style="padding: 10px;">DATE</th>
                        <th style="padding: 10px;">BUYS</th>
                        <th style="padding: 10px;">SELLS</th>
                        <th style="padding: 10px;">HOLDINGS</th>
                        <th style="padding: 10px;">DEMAT VALUE</th>
                        <th style="padding: 10px;">FREE CASH</th>
                        <th style="padding: 10px;">NET EQUITY</th>
                        <th style="padding: 10px;">DAY P/L</th>
                        <th style="padding: 10px;">CHARGES</th>
                    </tr>
                </thead>
                <tbody>
            ''')
            
            for idx, row in df_slice.iterrows():
                date_str = row['Date'].strftime("%d/%m/%Y")
                day_str = row['Date'].strftime("%a")
                
                # Buys
                buys_html = ""
                if isinstance(row.get('Buys_Raw'), list):
                    for b in row['Buys_Raw']:
                        buys_html += f"""
                        <div style="border: 1px solid #238636; border-radius: 6px; padding: 8px; margin-bottom: 4px; background: rgba(35, 134, 54, 0.1);">
                            <div style="font-weight: bold; margin-bottom: 4px; font-size: 13px;">{b['symbol']}</div>
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8b949e;">
                                <span>Buy Qty: {b['qty']}</span>
                                <span>Price: ₹{b['price']:.2f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8b949e;">
                                <span>Value: ₹{b['value']:.2f}</span>
                                <span>Charges: ₹{b['fee']:.2f}</span>
                            </div>
                        </div>
                        """
                
                # Sells
                sells_html = ""
                if isinstance(row.get('Sells_Raw'), list):
                    for s in row['Sells_Raw']:
                        s_color = "#3fb950" if s['pnl'] > 0 else "#ff7b72"
                        bg_color = "rgba(35, 134, 54, 0.1)" if s['pnl'] > 0 else "rgba(218, 54, 51, 0.1)"
                        border_color = "#238636" if s['pnl'] > 0 else "#da3633"
                        sells_html += f"""
                        <div style="border: 1px solid {border_color}; border-radius: 6px; padding: 8px; margin-bottom: 4px; background: {bg_color};">
                            <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 4px; font-size: 13px;">
                                <span>{s['symbol']}</span>
                                <span style="font-size: 10px; color: #c9d1d9; background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 4px;">{s['reason']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8b949e;">
                                <span>Sell Qty: {s['qty']}</span>
                                <span>Price: ₹{s['price']:.2f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8b949e;">
                                <span>Value: ₹{s['value']:.2f}</span>
                                <span style="color: {s_color}; font-weight:bold;">PnL: ₹{s['pnl']:.2f}</span>
                            </div>
                        </div>
                        """
                
                # Holdings
                holds = row.get('Holdings_Raw', [])
                if not isinstance(holds, list): holds = []
                
                if len(holds) > 0:
                    details_html = ""
                    for h in holds:
                        details_html += f"<div style='margin-bottom: 2px;'>{h['symbol']} ({h['qty']}) @ ₹{h['close']:.2f} <br><span style='color:#8b949e'>Inv: ₹{h['inv']:.2f}</span></div>"
                    holdings_html = f"""
                    <details>
                        <summary style='cursor: pointer; font-weight: bold; font-size: 13px; color: #58a6ff; outline: none;'>{len(holds)} holdings</summary>
                        <div style='margin-top: 8px; font-size: 11px; padding: 4px; background: rgba(0,0,0,0.2); border-radius: 4px;'>
                            {details_html}
                        </div>
                    </details>
                    """
                else:
                    holdings_html = f"<div style='font-weight: bold; font-size: 13px; color: #8b949e;'>0 holdings</div>"
                
                day_pl_color = "#3fb950" if row['Net Daily P&L'] >= 0 else "#ff7b72"
                
                html.append(f'''
                    <tr style="border-bottom: 1px solid #30363d; vertical-align: top;">
                        <td style="padding: 12px 10px;">
                            <div style="background: #1c2128; border-radius: 6px; padding: 8px; text-align: center; width: 80px;">
                                <div style="font-weight: bold; font-size: 14px; color: #c9d1d9;">{date_str}</div>
                                <div style="font-size: 11px; color: #8b949e;">{day_str}</div>
                            </div>
                        </td>
                        <td style="padding: 12px 10px; min-width: 180px;">{buys_html}</td>
                        <td style="padding: 12px 10px; min-width: 180px;">{sells_html}</td>
                        <td style="padding: 12px 10px;">{holdings_html}</td>
                        <td style="padding: 12px 10px;">
                            <div style="font-weight: bold; font-size: 13px; color: #c9d1d9;">₹{row['Deployed Capital']:,.2f}</div>
                            <div style="font-size: 11px; color: #8b949e;">inv</div>
                        </td>
                        <td style="padding: 12px 10px;">
                            <div style="font-weight: bold; font-size: 13px; color: #c9d1d9;">₹{row['Available Cash']:,.2f}</div>
                            <div style="font-size: 11px; color: #8b949e;">available</div>
                        </td>
                        <td style="padding: 12px 10px;">
                            <div style="font-weight: bold; font-size: 13px; color: #e0e0e0;">₹{row['Total Equity']:,.2f}</div>
                            <div style="font-size: 11px; color: #ff7b72;">DD {row['Drawdown %']:.2f}%</div>
                        </td>
                        <td style="padding: 12px 10px; font-weight: bold; font-size: 13px; color: {day_pl_color};">₹{row['Net Daily P&L']:,.2f}</td>
                        <td style="padding: 12px 10px;">
                            <div style="font-weight: bold; font-size: 13px; color: #c9d1d9;">₹{(row['Fees'] + row['Slippage']):,.2f}</div>
                            <div style="font-size: 11px; color: #8b949e;">fees + slip</div>
                        </td>
                    </tr>
                ''')
                
            html.append('</tbody></table></div>')
            return "".join(html)

        t1, t2, t3 = st.tabs(["Overview & KPIs", "Daily Ledger", "Trade Log"])
        with t1:
            st.markdown(render_kpis(kpis), unsafe_allow_html=True)
            st.plotly_chart(px.line(df_ledger, x='Date', y='Total Equity', title='Equity Curve'))
            st.plotly_chart(px.area(df_ledger, x='Date', y='Drawdown %', title='Drawdown %'))
            
            st.markdown("### Periodic Returns")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Yearly**")
                st.dataframe(df_yearly[['Year', 'Return %', 'Max_DD', 'Wins', 'Losses']], hide_index=True)
            with c2:
                st.write("**Monthly**")
                st.dataframe(df_monthly[['Month', 'Return %', 'Max_DD', 'Wins', 'Losses']], hide_index=True)
            with c3:
                st.write("**Weekly**")
                if not df_weekly.empty:
                    st.dataframe(df_weekly[['Week', 'Return %', 'Max_DD', 'Wins', 'Losses']].tail(100), hide_index=True)
                else:
                    st.write("No weekly data available.")
        
        with t2:
            components.html(render_custom_ledger(df_ledger), height=850, scrolling=True)
            
        with t3: 
            st.dataframe(df_trades)

    # --- Save Run UI ---
    st.markdown("---")
    st.subheader("Save Current Run")
    run_name = st.text_input("Run ID / Name", placeholder="e.g. Test V1 - 25% drop")
    if st.button("Save Run") and 'current_results' in st.session_state:
        if run_name.strip() == "":
            st.warning("Please enter a Run Name.")
        else:
            run_data = {
                'run_name': run_name,
                'params': st.session_state.current_results['params'],
                'kpis': st.session_state.current_results['kpis'],
                'results': st.session_state.current_results
            }
            st.session_state.saved_runs[run_name] = run_data
            
            safe_name = "".join([c for c in run_name if c.isalpha() or c.isdigit() or c in ' -_']).rstrip()
            filepath = os.path.join(SAVE_DIR, f"{safe_name}.pkl")
            with open(filepath, 'wb') as file:
                pickle.dump(run_data, file)
                
            st.success(f"Run '{run_name}' saved successfully! You can compare it in the Run Comparison tab.")

with tab_compare:
    st.subheader("Run Comparison")
    if not st.session_state.saved_runs:
        st.info("No saved runs yet. Run a backtest and save it to compare.")
    else:
        selected_runs = st.multiselect("Select Runs to Compare", list(st.session_state.saved_runs.keys()), default=list(st.session_state.saved_runs.keys()))
        if selected_runs:
            compare_data = []
            for run in selected_runs:
                row = {'Run ID': run}
                row.update(st.session_state.saved_runs[run]['kpis'])
                # Add key params for context
                p = st.session_state.saved_runs[run]['params']
                row['SMA'] = p['sma_period']
                row['Thresh %'] = p['mr_threshold_pct']
                row['Hold Days'] = p['hold_period']
                row['Alloc %'] = p['alloc_pct']
                row['Init SL'] = f"{p['initial_sl_pct']}%" if p['use_initial_sl'] else "OFF"
                row['Stages'] = len(p['enabled_stages'])
                compare_data.append(row)
                
            df_compare = pd.DataFrame(compare_data)
            st.dataframe(df_compare)
            
            # Bar chart comparison
            st.plotly_chart(px.bar(df_compare, x='Run ID', y='CAGR %', title="CAGR % Comparison", color='Run ID'))
            st.plotly_chart(px.bar(df_compare, x='Run ID', y='Max DD %', title="Max Drawdown % Comparison", color='Run ID'))
