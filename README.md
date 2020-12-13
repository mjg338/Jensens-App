# Jensen's Measure for S&P500 Securities

This application uses Jensen's Measure to assess the quality of individual stock picks in a user given time window using a given moving average. 

The data is sourced from yahoo finance, and is organized into several features: day, company, company stock ticker, sub-sector, sector, day average, 7 day average, 50 day average, 200 day average, weight. 

From this a plot can be generated for any given company on the S&P 500 using the given time window and moving average. This plot will produce a time series of share prices for the company selected, its sub-sector index, sector index, and an index of the S&P500 (VOO). This plot can serve as a useful visual to compare the performances of any of the 4 times series. 

The application will also produce a linear regression to quantify Jensen's Alpha and Beta. The linear regression produces a best fit based on a scatter of the percent returns for the stock and index within the given time frame. The slope of this best fit measures the volatility of the stock with respect to the index, and the intercept describes whether or not the returns of the stock were "worth" any additional risk taken. Even with a beta identifying considerably higher stock volality, a positive alpha suggests a quality stock choice.

Apart from showing the numerical values for alpha and beta, the metrics table at the bottom of the application can be used for assessing whether or not the regression is well fit to the data.

In instances where the regression score is less than 90%, it may be worth using a longer moving average to buff out some of the volatility and produce a model that better desribes the input data. Of course, these are individual stocks being used and not the diversified portfolios for which this measure is often used. Given that individual stocks will most often be more volatile that diverse portfolies, it is expected that there will be periods that are simply not amenable to regression.

Still, many periods are, meaning this application can be great for getting a quick assessment of a company's past stock performance.


Software used:

dash
pandas
numpy
matplotlib
datetime
sklearn
scipy
plotly
datetime
seaborn
statsmodels
