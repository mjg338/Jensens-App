import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import statsmodels.api as sm

df = pd.read_csv(r'sp500_final_6y_wMA_365.csv', sep= ',', index_col = 0)
df13 = pd.read_csv(r'VOO_data_6y_wMA_365.csv', sep= ',', index_col = 0)
df30 = pd.read_csv(r'return_free_rates.csv', sep= '\t')
df['Date'] = pd.to_datetime(df['Date'])
df13['Date'] = pd.to_datetime(df['Date'])
df30['Date'] = pd.to_datetime(df30['Date'])

df1 = df.groupby('Sector', as_index = False)["Close"].count()
df2 = df.groupby('Name', as_index = False)["Close"].count()
list1 = list(df1['Sector'])

df1 = df.groupby('Sub-Sector', as_index = False)["Close"].count()
list10 = list(df1['Sub-Sector'])

dict1 = {}
for i in list1:
    df3 = df[df['Sector'] == i]
    df4 = df3.groupby('Name', as_index = False)["Close"].count()
    list2 = list(df4['Name'])
    for f in list2:
        dict1[f] = i

dict2 = {}
for i in list10:
    df3 = df[df['Sub-Sector'] == i]
    df4 = df3.groupby('Name', as_index = False)["Close"].count()
    list2 = list(df4['Name'])
    for f in list2:
        dict2[f] = i

df1000 = df.groupby(['Name', 'Company'], as_index = False)["Day Average"].count()
list1000 = []
for index, row in df1000.iterrows():
    dict150 = {'label': '', 'value': ''}
    dict150['label'] = str(row.iloc[1])
    dict150['value'] = str(row.iloc[0])
    list1000.append(dict150)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

server = app.server

graph1 = dcc.Loading(
    dcc.Graph(id='figure-scatter'),
    type = 'default')

graph2 =     dcc.Loading(
    dcc.Graph(id='figure-regression',
             style=dict(border='2 solid black')),
    type = 'default')

table1 = dcc.Loading(
    dcc.Graph(id='figure-table'),
    type = 'default')

controls = [dcc.DatePickerRange(
        id = 'my-date',
        min_date_allowed=date(2015, 1, 1),
        max_date_allowed=date(2020, 11, 11),
        initial_visible_month=date(2017, 8, 5),
        start_date=date(2015, 1, 1),
        end_date=date(2020, 11, 1)),
            html.Br(),
           dcc.Dropdown(id = 'MA',
    options=[
        {'label': 'Day Average', 'value': 'Day Average'},
        {'label': '7 Day Avg', 'value': '7 Day Avg'},
        {'label': '50 Day Avg', 'value': '50 Day Avg'},
        {'label': '200 Day Avg', 'value': '200 Day Avg'},
        {'label': '365 Day Avg', 'value': '365 Day Avg'}
    ], value='Day Average'),
            html.Br(), 
            dcc.Dropdown(id = 'Name',
    options = list1000,
    value='GOOG'),
            html.Br(),
            
           
           html.Button('Submit', id='submit-val', n_clicks=0)]


app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Card(controls, body=True)
                ], width=4),
                dbc.Col([
                    dbc.Card(graph1, body=False)
                ], width=8)
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(graph2, body=False) 
                ], width=8),
                dbc.Col([
                    dbc.Card(table1, body=False)
                ], width=4)
            ], align='center'), 
            html.Br(),    
        ]), color = 'dark'
    )
])

@app.callback(Output(component_id='figure-scatter', component_property='figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State("Name", "value"),
     dash.dependencies.State("MA", "value"),
     dash.dependencies.State("my-date", "start_date"),
     dash.dependencies.State("my-date", "end_date")] )
def compare_to_market1(Button, Name, MA, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m-%d-%y')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m-%d-%y')
    
    df1 = df[(df.Date >= start_date) & (df.Date <= end_date)]
    df131 = df13[(df13.Date >= start_date) & (df13.Date <= end_date)]
    
    #Build Sector plot
    sector = dict1[Name]
    df5 = df1[df1['Sector'] == sector]
    df6 = df5.groupby('Date', as_index = False)[MA].sum()
    list2 = []
    for index, row in df6.iterrows():
        df7 = df5[df5['Date'] == row[0]]
        list3 = list(df7['Name'])
        list2.append(len(list3))
    df6['Comp in Sector'] = list2
    df6['Sector Average'] = df6[MA]/df6['Comp in Sector']
    
    #Build Sub-Sector plot
    subsector = dict2[Name]
    df10 = df1[df1['Sub-Sector'] == subsector]
    df11 = df10.groupby('Date', as_index = False)[MA].sum()
    list4 = []
    for index, row in df11.iterrows():
        df12 = df10[df10['Date'] == row[0]]
        list5 = list(df12['Name'])
        list4.append(len(list5))
    df11['Comp in Sub-Sector'] = list4
    df11['Sub-Sector Average'] = df11[MA]/df11['Comp in Sub-Sector']
    
    #Build Company plot
    df8 = df1[df1['Name'] == Name]
    df9 = pd.DataFrame()
    df9['Date'] = df8['Date']
    df9[MA] = df8[MA]
    df9 = df9.sort_values('Date')
    Company = df8.iloc[1,11]
    
    
    #Plot with Plotly
    df50 = pd.DataFrame()
    df50['Date'] = df11['Date']
    df50['S&P Average'] = np.array(df131[MA])
    df50['Sector Average'] = df6['Sector Average']
    df50['Sub-Sector Average'] = df11['Sub-Sector Average']
    df50[Name] = np.array(df9[MA])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df50['Date'], y=df50['S&P Average'],
                    mode='lines',
                    name='S&P 500 (VOO)'))
    fig.add_trace(go.Scatter(x=df50['Date'], y=df50['Sector Average'],
                    mode='lines',
                    name=str(sector)))
    fig.add_trace(go.Scatter(x=df50['Date'], y=df50['Sub-Sector Average'],
                    mode='lines',
                    name=str(subsector)))
    fig.add_trace(go.Scatter(x=df50['Date'], y=df50[Name],
                    mode='lines',
                    name= '{} ({})'.format(Company, Name)))
    fig.update_layout(
        title="Index, Sector, and Sub-Sector for {} using {}".format(Company, MA),
        template='plotly_dark',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        yaxis_title="Share Price ($)",
        font=dict(
                size=12,
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor='lightgrey')
    fig.update_xaxes(showline=True, gridwidth=0.25, linecolor='lightgrey', mirror = True)
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, gridwidth=0.25, linecolor='lightgrey', mirror = True)
    
    iplot = {'data': fig.data,
             'layout': fig.layout}
    
    return iplot

@app.callback(Output(component_id='figure-regression', component_property='figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State("Name", "value"),
     dash.dependencies.State("MA", "value"),
     dash.dependencies.State("my-date", "start_date"),
     dash.dependencies.State("my-date", "end_date")] )
def compare_to_market2(Button, Name, MA, start_date, end_date):
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m-%d-%y')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m-%d-%y')
    
    df1 = df[(df.Date >= start_date) & (df.Date <= end_date)]
    df131 = df13[(df13.Date >= start_date) & (df13.Date <= end_date)]
    
    #Build Company plot
    df8 = df1[df1['Name'] == Name]
    df9 = pd.DataFrame()
    df9['Date'] = df8['Date']
    df9[MA] = df8[MA]
    df9 = df9.sort_values('Date')
    Company = df8.iloc[1,11]

    
    
    #Find Date for Risk Free Rate
    FMT = '%m-%d-%y'
    tdelta = datetime.strptime(end_date, FMT) - datetime.strptime(start_date, FMT)


    date = datetime.strptime(start_date, '%m-%d-%y')
    date1 = datetime.strftime(date, '%m-%d-%y')
    try:
        df31 = df30[df30['Date'] == str(date1)]
        rfm = float(df31['1 Mo'])
        use_date = str(date1)
    except:
        try:
            date2 = datetime.strptime(date1, '%m-%d-%y')
            date3 = date2 + timedelta(days=1)
            date4 = datetime.strftime(date3, '%m-%d-%y')
            df32 = df30[df30['Date'] == str(date4)]
            rfm = float(df32['1 Mo'])
            use_date = str(date4)
        except:
            try:
                date5 = datetime.strptime(date4, '%m-%d-%y')
                date6 = date5 + timedelta(days=1)
                date7 = datetime.strftime(date6, '%m-%d-%y')
                df33 = df30[df30['Date'] == str(date7)]
                rfm = float(df33['1 Mo'])
                use_date = str(date7)
            except:
                try:
                    date8 = datetime.strptime(date7, '%m-%d-%y')
                    date9 = date8 + timedelta(days=1)
                    date10 = datetime.strftime(date9, '%m-%d-%y')
                    df33 = df30[df30['Date'] == str(date10)]
                    rfm = float(df33['1 Mo'])
                    use_date = str(date10)
                except:
                    print('error')

    #Find risk free rate
    df31 = df30[df30['Date'] == use_date]

    if tdelta.days <= 30:
        rf = float(df31['1 Mo'])
    elif tdelta.days > 30 and tdelta.days <= 90:
        rf = float(df31['3 Mo'])
    elif tdelta.days > 90 and tdelta.days <= 180:
        rf = float(df31['6 Mo'])
    elif tdelta.days > 180 and tdelta.days <= 365:
        rf = float(df31['1 Yr'])
    elif tdelta.days > 365 and tdelta.days <= 730:
        rf = float(df31['2 Yr'])
    elif tdelta.days > 730 and tdelta.days <= 1105:
        rf = float(df31['3 Yr'])
    elif tdelta.days > 1105 and tdelta.days <= 1835:
        rf = float(df31['5 Yr'])
    elif tdelta.days > 1835 and tdelta.days <= 2565:
        rf = float(df31['7 Yr'])


    #Develop Regression
    list20 = list(df8[MA])
    list22 = list(df131[MA])
    list21 = []
    list23 = []
    for i in list20[1:]:
        x = round((100*(i/list20[0] - 1) - rf), 3)
        list21.append(x)
    for i in list22[1:]:
        x = round((100*(i/list22[0] - 1) - rf), 3)
        list23.append(x)
    
    myarray1 = np.asarray(list21)
    myarray2 = np.asarray(list23)
    
    df60 = pd.DataFrame()
    df60['X'] = np.asarray(list23)
    df60['Y'] = np.asarray(list21)
    df60



    x = myarray2.reshape(-1, 1)
    y = myarray1.reshape(-1, 1)

    X = df60.X[:, None]
    X_train, X_test, y_train, y_test = train_test_split(X, df60.Y, random_state=0)

    model = LinearRegression()
    model.fit(X, df60.Y)

    y_pred = model.predict(X_test)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    
    
    #Find relevant metrics
    mean = round((sum(list21)/len(list21)), 3)
    alpha = round(float(model.intercept_), 3)
    beta = round(float(model.coef_), 3)
    error = round(float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))), 3)
    discriminator = error/mean
    disc = round((100*(discriminator)), 3)
    disc_perc = str(disc) + '%'
    

    #Return regression analytics if applicable
    #if discriminator <= float(err) and discriminator >= -float(err):
    fig1 = go.Figure([go.Scatter(x = df60.X.squeeze(), y = df60.Y, name='Returns', mode='markers'),
    go.Scatter(x=x_range, y=y_range, name='Fit')
    ])
    fig1.update_layout(
        title="Linear Regression of Percent Returns for {} and S&P 500".format(Company),
        xaxis_title="S&P 500 (VOO) - RFR",
        template='plotly_dark',
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        yaxis_title="{} - RFR".format(Company + ' ({})'.format(Name)),
        font=dict(
                size=12,
        )
    )
    fig1.update_xaxes(showline=True, linewidth=.25, linecolor='lightgrey', mirror = True, showgrid=True, gridwidth=0.25, gridcolor='lightgrey')
    fig1.update_xaxes(zeroline=True, zerolinewidth=.75, zerolinecolor='lightgrey')
    fig1.update_yaxes(showline=True, linewidth=.25, linecolor='lightgrey', mirror = True, showgrid=True, gridwidth=0.25, gridcolor='lightgrey')
    fig1.update_yaxes(zeroline=True, zerolinewidth=.75, zerolinecolor='lightgrey')

    
    iplot = {'data': fig1.data,
             'layout': fig1.layout}

    return iplot

@app.callback(Output(component_id='figure-table', component_property='figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State("Name", "value"),
     dash.dependencies.State("MA", "value"),
     dash.dependencies.State("my-date", "start_date"),
     dash.dependencies.State("my-date", "end_date")] )
def compare_to_market3(Button, Name, MA, start_date, end_date):
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m-%d-%y')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m-%d-%y')
    
    df1 = df[(df.Date >= start_date) & (df.Date <= end_date)]
    df131 = df13[(df13.Date >= start_date) & (df13.Date <= end_date)]
    
    #Build Company plot
    df8 = df1[df1['Name'] == Name]
    df9 = pd.DataFrame()
    df9['Date'] = df8['Date']
    df9[MA] = df8[MA]
    df9 = df9.sort_values('Date')
    Company = df8.iloc[1,11]

    
    
    #Find Date for Risk Free Rate
    FMT = '%m-%d-%y'
    tdelta = datetime.strptime(end_date, FMT) - datetime.strptime(start_date, FMT)


    date = datetime.strptime(start_date, '%m-%d-%y')
    date1 = datetime.strftime(date, '%m-%d-%y')
    try:
        df31 = df30[df30['Date'] == str(date1)]
        rfm = float(df31['1 Mo'])
        use_date = str(date1)
    except:
        try:
            date2 = datetime.strptime(date1, '%m-%d-%y')
            date3 = date2 + timedelta(days=1)
            date4 = datetime.strftime(date3, '%m-%d-%y')
            df32 = df30[df30['Date'] == str(date4)]
            rfm = float(df32['1 Mo'])
            use_date = str(date4)
        except:
            try:
                date5 = datetime.strptime(date4, '%m-%d-%y')
                date6 = date5 + timedelta(days=1)
                date7 = datetime.strftime(date6, '%m-%d-%y')
                df33 = df30[df30['Date'] == str(date7)]
                rfm = float(df33['1 Mo'])
                use_date = str(date7)
            except:
                try:
                    date8 = datetime.strptime(date7, '%m-%d-%y')
                    date9 = date8 + timedelta(days=1)
                    date10 = datetime.strftime(date9, '%m-%d-%y')
                    df33 = df30[df30['Date'] == str(date10)]
                    rfm = float(df33['1 Mo'])
                    use_date = str(date10)
                except:
                    print('error')

    #Find risk free rate
    df31 = df30[df30['Date'] == use_date]

    if tdelta.days <= 30:
        rf = float(df31['1 Mo'])
    elif tdelta.days > 30 and tdelta.days <= 90:
        rf = float(df31['3 Mo'])
    elif tdelta.days > 90 and tdelta.days <= 180:
        rf = float(df31['6 Mo'])
    elif tdelta.days > 180 and tdelta.days <= 365:
        rf = float(df31['1 Yr'])
    elif tdelta.days > 365 and tdelta.days <= 730:
        rf = float(df31['2 Yr'])
    elif tdelta.days > 730 and tdelta.days <= 1105:
        rf = float(df31['3 Yr'])
    elif tdelta.days > 1105 and tdelta.days <= 1835:
        rf = float(df31['5 Yr'])
    elif tdelta.days > 1835 and tdelta.days <= 2565:
        rf = float(df31['7 Yr'])


    #Develop Regression
    list20 = list(df8[MA])
    list22 = list(df131[MA])
    list21 = []
    list23 = []
    for i in list20[1:]:
        x = round((100*(i/list20[0] - 1) - rf), 3)
        list21.append(x)
    for i in list22[1:]:
        x = round((100*(i/list22[0] - 1) - rf), 3)
        list23.append(x)
    
    df60 = pd.DataFrame()
    df60['X'] = np.asarray(list23)
    df60['Y'] = np.asarray(list21)
    df60
    
    myarray1 = np.asarray(list21)
    myarray2 = np.asarray(list23)
   

    x = myarray2.reshape(-1, 1)
    y = myarray1.reshape(-1, 1)

    X = df60.X[:, None]
    X_train, X_test, y_train, y_test = train_test_split(X, df60.Y)
    
    model = LinearRegression()
    model.fit(X, df60.Y)
    
    y_pred = model.predict(X_test)
    
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    
    N = len(list21)
    
    Rsq = model.score(X, df60.Y)
    
    
    #Find relevant metrics
    mean = round((sum(list21)/len(list21)), 3)
    alpha = round(float(model.intercept_), 3)
    beta = round(float(model.coef_), 3)
    error = round(float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))), 3)
    discriminator = error/mean
    disc = round((100*(discriminator)), 3)
    disc_perc = str(disc) + '%'
    score = 100*(round((model.score(X, df60.Y)), 3))
    fstat = (round((Rsq/(1-Rsq))*((N-2-1)/2)), 3)
    
    fig2 = go.Figure(data=[go.Table(header=dict(values=['Regression Metric', 'Values (%)']),
         cells=dict(values=[['Risk-Free Rate (RFR)', 'Alpha', 'Beta', 'R-Squared', 'Root Mean Squared Error'], [str(rf), alpha, beta, round(score, 3), error]]))
             ])
    fig2.update_layout(template='plotly_dark', plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)')

    iplot = {'data': fig2.data,
             'layout': fig2.layout}

    return iplot


if __name__ == '__main__':
    app.run_server(debug=False)