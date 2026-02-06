from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yfi
from dash import Dash, html, callback, Output, Input, dcc
import dash_bootstrap_components as dbc
from sklearn.linear_model import ElasticNet
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Try to import the C++ extension
try:
    import forecast_cpp
    USE_CPP = True
    print("C++ extension loaded successfully!")
except ImportError:
    USE_CPP = False
    print("C++ extension not found, using Python fallback.")

def warn(*args, **kwargs):
    pass
warnings.warn = warn

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

### -------------------------------------------------------- ###
### -----------------EXTRA STUFF - LUKE -------------------- ###
### -------------------------------------------------------- ###
stock_list = ['AAPL','AMZN','NVDA','MSFT','IBM','INTC']

slider_date_yfi_dict = {0:'1d' ,1:'1wk',2:'1mo',
                    3:'6mo',4:'1y',5:'5y'}
slider_date_yfi_interval_dict = {0:['5m','1 Day'],1:['30m','1 Week'],
                                 2:['30m','1 Month'],3:['1h','6 Months'],
                                 4:['1d','1 Year'],5:['1d','5 Years']}

radio_yfi_list = ['Open Price','Volume of Shares Sold','Gross Profit'] ### THIS IS MY BUTTON NAMES
radio_yfi_data_dict = {'Open Price':'Open', 'Volume of Shares Sold':'Volume'}

### -------------------------------------------------------- ###
### -----------------EXTRA STUFF - DURGESH ----------------- ###
### -------------------------------------------------------- ###

radio_yfi_list_tab2 = ['None','ElasticNet', 'ARIMA', 'Kalman Filter (HFT)', 'Monte Carlo (HFT)'] ### THIS IS MY BUTTON NAMES

### -------------------------------------------------------- ###
### -----------------EXTRA STUFF - MICHAEL------------------ ###
### -------------------------------------------------------- ###

radio_yfi_list_tab3 = ['Open Price','Volume of Shares Sold'] ### THIS IS MY BUTTON NAMES


app.layout = dbc.Container([
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0), # Auto-refresh every 60s
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label="Tab 1 - Stock Data Visualizations", children = [
            html.Div([
            html.H1(children='Stock data visualization', style={'textAlign':'center', 'marginBottom': '20px', 'marginTop': '20px'}),
            dbc.Input(id='DROPDOWN-SELECTION', value='AMZN', type='text', debounce=True, 
                     style={'color': 'black', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'},
                     placeholder="Enter Stock Ticker (e.g. TSLA, GOOGL)"),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Graph(id='GRAPH-CONTENT',responsive=True, style={'height': '60vh'}), width=9),  
                dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4("Select Metric", className="card-title", style={'color': 'white', 'textAlign': 'center'}),
                                dcc.RadioItems(options = radio_yfi_list, value = 'Open Price',id = 'RADIO-SELECTION',
                                        style={'textAlign':'left', 'font-size':18, 'color': 'white', 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'},
                                        inputStyle={"margin-right": "10px"})
                            ])
                        ], color="secondary", outline=True, style={'height': '100%'}), width=3)       
                    ], className="g-3", align="stretch"),
            html.Br(),
            dcc.Slider(0,5,step=None,marks=slider_date_yfi_dict, value=0,id='SLIDER-SELECTION')
                    ]),

        dcc.Tab(label='Tab 2 - Stock price forecasting',
               children = [
                html.Div([
                html.H1(children='Stock Data Forecasting', style={'textAlign':'center', 'marginBottom': '20px', 'marginTop': '20px'}),
                dbc.Input(id='DROPDOWN-SELECTION-2', value='AMZN', type='text', debounce=True, 
                     style={'color': 'black', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'},
                     placeholder="Enter Stock Ticker (e.g. TSLA, GOOGL)"),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='GRAPH-CONTENT-2',responsive=True, style={'height': '60vh'}), width=9),  
                    dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4("Forecasting Model", className="card-title", style={'color': 'white', 'textAlign': 'center'}),
                                dcc.RadioItems(options = radio_yfi_list_tab2, value = 'None',id = 'RADIO-SELECTION-2',
                                        style={'textAlign':'left', 'font-size':18, 'color': 'white', 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'},
                                        inputStyle={"margin-right": "10px"}),
                                html.Hr(),
                                html.Label("Forecast Days:", style={'textAlign':'center', 'width': '100%', 'color': 'white', 'fontSize': 18}),
                                 dcc.Input(value=1, type='number', min=1, id = 'K-SELECTION-2', style={'width': '100%', 'color': 'black'})
                             ])
                         ], color="secondary", outline=True, style={'height': '100%'}), width=3)
                    ], className="g-3", align="stretch") 
               ]),

        dcc.Tab(label='Tab 3 - Stock price decomposition', children=[
            html.Div([
                html.H1(children='Stock Data Decomposition', style={'textAlign':'center', 'marginBottom': '20px', 'marginTop': '20px'}),
                dbc.Input(id='DROPDOWN-SELECTION-3', value='AMZN', type='text', debounce=True, 
                     style={'color': 'black', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'},
                     placeholder="Enter Stock Ticker (e.g. TSLA, GOOGL)"),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='GRAPH-CONTENT-3',responsive=True, style={'height': '60vh'}), width=9),  
                    dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4("Decomposition Metric", className="card-title", style={'color': 'white', 'textAlign': 'center'}),
                                dcc.RadioItems(options = radio_yfi_list_tab3, value = 'Open Price',id = 'RADIO-SELECTION-3',
                                        style={'textAlign':'left', 'font-size':18, 'color': 'white', 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'},
                                        inputStyle={"margin-right": "10px"})
                            ])
                        ], color="secondary", outline=True, style={'height': '100%'}), width=3)
                    ], className="g-3", align="stretch"),
            html.Br(),
            dcc.Slider(0,5,step=None,marks=slider_date_yfi_dict, value=0,id='SLIDER-SELECTION-3')
                    ]),
               ])
    ], fluid=True)


### Function to update the graph on tab 1
@callback(
    Output('GRAPH-CONTENT', 'figure'),
    Input('DROPDOWN-SELECTION', 'value'),
    Input('SLIDER-SELECTION', 'value'),
    Input('RADIO-SELECTION','value'),
    Input('interval-component', 'n_intervals')
)
def update_graph(dropdown_selection, slider_selection, radio_selection, n_intervals):
    if not dropdown_selection:
        return go.Figure()
        
    if radio_selection in ['Open Price','Volume of Shares Sold']:
        col_name = radio_yfi_data_dict[radio_selection]
        try:
            dff = yfi.Ticker(dropdown_selection).history(period = slider_date_yfi_dict[slider_selection],
                                                interval=slider_date_yfi_interval_dict[slider_selection][0])
            if dff.empty:
                return go.Figure()
                
            fig = px.line(dff, x=dff.index, y=col_name, title=f"{dropdown_selection} - {slider_date_yfi_interval_dict[slider_selection][1]}")
            fig.update_layout(template='plotly_dark', yaxis_title=radio_selection)
            return fig
        except Exception:
            return go.Figure()
            
    if radio_selection == 'Gross Profit':
        try:
            dff = yfi.Ticker(dropdown_selection).financials.loc['Gross Profit']
            fig = px.line(x=dff.index, y=dff.values, title=f"{dropdown_selection} - Gross Profit ($)")
            fig.update_layout(template='plotly_dark', yaxis_title='Gross Profit ($)', xaxis_title='Date')
            return fig
        except Exception:
            return go.Figure()
    return go.Figure()

### Function to update the graph on tab 2
@callback(
    Output('GRAPH-CONTENT-2', 'figure'),
    Input('DROPDOWN-SELECTION-2', 'value'),
    Input('RADIO-SELECTION-2','value'),
    Input('K-SELECTION-2', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_graph2(dropdown_selection2, radio_selection2, h_selection2, n_intervals):
    if h_selection2 is None or int(h_selection2) < 1:
        return go.Figure()
    
    h_selection2 = int(h_selection2)
    
    if not dropdown_selection2:
        return go.Figure()
        
    try:
        data_open = pd.DataFrame(yfi.Ticker(dropdown_selection2).history(period='1y', interval='1d')['Open'])
        if data_open.empty:
            return go.Figure()
    except Exception:
        return go.Figure()
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_open.index, y=data_open['Open'], mode='lines', name='Historical Share Price'))
        
    title = f"{dropdown_selection2} - 1 year"
    textstr = ""

    if radio_selection2 == 'None':
         pass # Just historical

    elif radio_selection2 in ['ElasticNet','Both']:
        data_open_en = data_open.copy()
        for i in range(1, 11):
            data_open_en[f'Open_{i}day_delay'] = data_open_en['Open'].shift(i)
    
        data_open_en = data_open_en.dropna()
        train_df_x = data_open_en.drop(['Open'],axis=1)
        train_df_y = data_open_en['Open']
    
        model = ElasticNet(alpha=1, l1_ratio = 0.12)
        model.fit(X=train_df_x, y=train_df_y)

        ### Window for prediction
        pred_list_en = []
        
        # Initial input window [T, T-1, ..., T-9]
        recent_data = list(data_open['Open'].values[-10:]) 
        recent_data.reverse() 
        
        if USE_CPP:
            # Use C++ extension for speed
            # Pass coefficients, intercept, initial window, and steps
            # Note: model.coef_ in sklearn is usually in order of features X.
            # Our features X are [Open_1day_delay, ..., Open_10day_delay]
            # Open_1day_delay corresponds to T-1 (most recent).
            # recent_data is [T, T-1, ..., T-9].
            # Wait, let's double check the feature definition in python loop above:
            # data_open_en[f'Open_{i}day_delay'] = data_open_en['Open'].shift(i) for i in 1..10
            # shift(1) is previous day.
            # So at time T, features are [T-1, T-2, ..., T-10]
            # To predict T+1, we need features [T, T-1, ..., T-9]
            # recent_data is [T, T-1, ..., T-9] (reversed from original time series)
            # So the order matches.
            
            preds = forecast_cpp.recursive_forecast(
                model.coef_.tolist(),
                model.intercept_,
                recent_data,
                h_selection2
            )
        else:
            # Python fallback
            current_input = np.array(recent_data).reshape(1, -1)
            preds = []
            for _ in range(h_selection2):
                pred = model.predict(current_input)[0]
                preds.append(pred)
                
                # Update input for next step
                # shift right and insert new pred at 0
                current_input = np.insert(current_input[:, :-1], 0, pred).reshape(1, -1)
        
        pred_dates = pd.date_range(data_open.index[-1] + pd.Timedelta(days=1), periods=h_selection2)
        pred_series_en = pd.Series(preds, index=pred_dates)
        
        # Connect the lines
        connect_x = [data_open.index[-1], pred_dates[0]]
        connect_y = [data_open['Open'].iloc[-1], preds[0]]
        fig.add_trace(go.Scatter(x=connect_x, y=connect_y, mode='lines', line=dict(color='orange', dash='dash'), showlegend=False))
        
        fig.add_trace(go.Scatter(x=pred_series_en.index, y=pred_series_en, mode='lines', name='ElasticNet Forecast', line=dict(color='orange')))

        en_diff = (preds[-1] - data_open['Open'].iloc[-1]) / data_open['Open'].iloc[-1] * 100
        en_diff = np.round(en_diff, 2)
        if 'ElasticNet' in textstr:
            textstr += f"\nElasticNet predicted difference: {en_diff}%"
        else:
            textstr += f"ElasticNet predicted difference: {en_diff}%"

    if radio_selection2 in ['ARIMA', 'Both']:
        # Simplified ARIMA (Autoregressive only) for speed
        # Using order (5,1,0) which is essentially AR(5) on differenced data
        # This is much faster than auto_arima
        try:
            model_arima = ARIMA(data_open['Open'], order=(5,1,0))
            model_fit = model_arima.fit()
            forecast_res = model_fit.forecast(steps=h_selection2)
            
            pred_series_arima = pd.Series(forecast_res.values, index=pd.date_range(data_open.index[-1] + pd.Timedelta(days=1), periods=h_selection2))
            
            # Connect the lines
            connect_x = [data_open.index[-1], pred_series_arima.index[0]]
            connect_y = [data_open['Open'].iloc[-1], pred_series_arima.iloc[0]]
            fig.add_trace(go.Scatter(x=connect_x, y=connect_y, mode='lines', line=dict(color='green', dash='dash'), showlegend=False))

            fig.add_trace(go.Scatter(x=pred_series_arima.index, y=pred_series_arima, mode='lines', name='ARIMA Forecast', line=dict(color='green')))
            
            arima_diff = (pred_series_arima.iloc[-1] - data_open['Open'].iloc[-1]) / data_open['Open'].iloc[-1] * 100
            arima_diff = np.round(arima_diff, 2)
            
            if textstr:
                textstr += f"<br>ARIMA predicted difference: {arima_diff}%"
            else:
                textstr += f"ARIMA predicted difference: {arima_diff}%"
        except Exception as e:
            print(f"ARIMA Error: {e}")

    if radio_selection2 == 'Kalman Filter (HFT)':
        # HFT-style Kalman Filter Forecast
        if USE_CPP:
             # Use fast C++ implementation
             # Get all data as list
             full_history = data_open['Open'].values.tolist()
             
             preds_kalman = forecast_cpp.kalman_forecast(
                 full_history,
                 h_selection2,
                 1e-5, # Q (Process noise - tunable)
                 1e-3  # R (Measurement noise - tunable)
             )
             
             pred_dates_kalman = pd.date_range(data_open.index[-1] + pd.Timedelta(days=1), periods=h_selection2)
             pred_series_kalman = pd.Series(preds_kalman, index=pred_dates_kalman)
             
             # Connect
             connect_x = [data_open.index[-1], pred_series_kalman.index[0]]
             connect_y = [data_open['Open'].iloc[-1], pred_series_kalman.iloc[0]]
             fig.add_trace(go.Scatter(x=connect_x, y=connect_y, mode='lines', line=dict(color='cyan', dash='dash'), showlegend=False))
             
             fig.add_trace(go.Scatter(x=pred_series_kalman.index, y=pred_series_kalman, mode='lines', name='Kalman Filter Forecast', line=dict(color='cyan')))
             
             kalman_diff = (pred_series_kalman.iloc[-1] - data_open['Open'].iloc[-1]) / data_open['Open'].iloc[-1] * 100
             kalman_diff = np.round(kalman_diff, 2)
             
             if textstr:
                textstr += f"<br>Kalman Filter diff: {kalman_diff}%"
             else:
                textstr += f"Kalman Filter diff: {kalman_diff}%"
        else:
             if textstr:
                textstr += "<br>C++ extension required for Kalman Filter"
             else:
                textstr = "C++ extension required for Kalman Filter"

    if radio_selection2 == 'Monte Carlo (HFT)':
        # Monte Carlo Simulation (GBM)
        if USE_CPP:
             full_history = data_open['Open'].values.tolist()
             
             # Simulate 1000 paths and get the average
             preds_mc = forecast_cpp.monte_carlo_forecast(
                 full_history,
                 h_selection2,
                 1000 # Number of simulations
             )
             
             pred_dates_mc = pd.date_range(data_open.index[-1] + pd.Timedelta(days=1), periods=h_selection2)
             pred_series_mc = pd.Series(preds_mc, index=pred_dates_mc)
             
             # Connect
             connect_x = [data_open.index[-1], pred_series_mc.index[0]]
             connect_y = [data_open['Open'].iloc[-1], pred_series_mc.iloc[0]]
             fig.add_trace(go.Scatter(x=connect_x, y=connect_y, mode='lines', line=dict(color='magenta', dash='dash'), showlegend=False))
             
             fig.add_trace(go.Scatter(x=pred_series_mc.index, y=pred_series_mc, mode='lines', name='Monte Carlo (GBM) Forecast', line=dict(color='magenta')))
             
             mc_diff = (pred_series_mc.iloc[-1] - data_open['Open'].iloc[-1]) / data_open['Open'].iloc[-1] * 100
             mc_diff = np.round(mc_diff, 2)
             
             if textstr:
                textstr += f"<br>Monte Carlo diff: {mc_diff}%"
             else:
                textstr += f"Monte Carlo diff: {mc_diff}%"
        else:
             if textstr:
                textstr += "<br>C++ extension required for Monte Carlo"
             else:
                textstr = "C++ extension required for Monte Carlo"


    fig.update_layout(template='plotly_dark', title=title, yaxis_title='Share Price')
    
    if textstr:
         fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=textstr,
            showarrow=False,
            bgcolor="wheat",
            opacity=0.8,
            font=dict(color="black")
        )

    return fig
    return go.Figure()


### Function to update the graph on tab 3
@callback(
    Output('GRAPH-CONTENT-3', 'figure'),
    Input('DROPDOWN-SELECTION-3', 'value'),
    Input('SLIDER-SELECTION-3', 'value'),
    Input('RADIO-SELECTION-3','value'),
    Input('interval-component', 'n_intervals')
)
def update_graph_3(dropdown_selection3, slider_selection3, radio_selection3, n_intervals):
    if not dropdown_selection3:
        return go.Figure()
        
    col_name = radio_yfi_data_dict[radio_selection3]
    try:
        dff = yfi.Ticker(dropdown_selection3).history(period = slider_date_yfi_dict[slider_selection3],
                                            interval=slider_date_yfi_interval_dict[slider_selection3][0])
        if dff.empty:
            return go.Figure()

        period = 12 # default
        model_type = 'multiplicative'
        
        # Mapping logic from original
        if slider_selection3 == 5:
            period = 250
        elif slider_selection3 == 4:
            period = 12
        elif slider_selection3 == 3:
            if col_name == 'Open': period = 125
            else: 
                model_type = 'additive'
                period = 65
        elif slider_selection3 == 2:
            period = 65
        elif slider_selection3 == 1:
            period = 12
        elif slider_selection3 == 0:
            period = 12
            if col_name == 'Volume': model_type = 'additive'

        # Ensure no zero/negative values for multiplicative
        if model_type == 'multiplicative' and (dff[col_name] <= 0).any():
             model_type = 'additive'

        try:
            result = seasonal_decompose(dff[col_name], model=model_type, period=period)
            
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
            
            fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name='Observed'), row=1, col=1)
            fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name='Residual'), row=4, col=1)
            
            title_str = f"{dropdown_selection3} - {slider_date_yfi_interval_dict[slider_selection3][1]} Decomposition"
            fig.update_layout(height=800, template='plotly_dark', title=title_str, showlegend=False)
            
            return fig
        except Exception as e:
            # Fallback if decomposition fails (e.g. not enough data)
            fig = go.Figure()
            fig.update_layout(title=f"Error performing decomposition: {str(e)}", template='plotly_dark')
            return fig
    except Exception:
        return go.Figure()

    return go.Figure()

if __name__ == '__main__':
    app.run(debug=True)
