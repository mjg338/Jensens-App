# Jensen's Measure for S&P500 Securities

This application uses Jensen's Measure to assess the volatility and risk adjusted returns of any given company on the S&P 500. A video explaining the application and its uses is given here:

https://youtu.be/NieKFjTBZ1c

The csv files, which are too large for an upload to github, can be downloaded here:

https://drive.google.com/drive/folders/1jUeswnuTn4Rcpqkp4BtBrvQ5szKMdKRM?usp=sharing


Based on the research I've done, I would not suggest people buy securities solely on the basis of high alpha scores, only that the share values of these companies are consistently outgrowing most of the public market in the chosen time period. The decision to purchase a selection of individual stocks with high risk-adjusted returns (i.e. high alpha scores) is a choice that should be based on a number of other factors as well. 

Other things to be considered would be things like financial resilience, as in, how much volatility is an owner willing and able to take on? Company stock growth can also be artificial, as in the growth of its stock might be based on an influx of buyers that is not comensurate to its worth. These are things that cannot be captured by the application to inform a decision.


Packages used:

dash,
pandas,
numpy,
matplotlib,
datetime,
sklearn,
scipy,
plotly,
datetime,
seaborn,
statsmodels


App uploaded using heroku and gunicorn:

https://jensens-app.herokuapp.com

If you have trouble loading, try a refresh or coming back to the app after a few minutes. If the app hasnt been used by anyone recently, it will take 5-10 minutes for the dyno running the app to boot up. Once its going the app should run smoothly.
