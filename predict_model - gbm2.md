# Statistical analysis of prices prediction

Under the conditions of the example built in this repository, it is known that the material price is based on the stock value of the scrap steel.

The Geometric Browninan Motion discret-model is one of the most common used methods to assess of future value of metal prices. Then, it is possible to assess a price prediction analysis based on probability distributions.

## Prepare data

In the web exist numerous sites where is possible to retrieve stock prices. Even python packages such as YahooFinancials can help Data Scientist to get access to a large amount of financial data.

In this case, I had acces of a previuos data set that corresponds to the scrap steel value in the last 5 years.


```python
import pandas as pd

data = pd.read_excel("COSTOS.xlsx",'BASE2',usecols='A:G')
new = pd.read_excel("COSTOS.xlsx",'BASE2_NEW',usecols='A:G')

```


```python
data
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FECHA</th>
      <th>ULT</th>
      <th>APT</th>
      <th>MAX</th>
      <th>MIN</th>
      <th>VOL</th>
      <th>VAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-11-24</td>
      <td>195.00</td>
      <td>195.00</td>
      <td>195.00</td>
      <td>195.00</td>
      <td>-</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-25</td>
      <td>195.00</td>
      <td>195.00</td>
      <td>195.00</td>
      <td>195.00</td>
      <td>-</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-11-26</td>
      <td>192.00</td>
      <td>192.00</td>
      <td>192.00</td>
      <td>192.00</td>
      <td>-</td>
      <td>-0.0154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-11-27</td>
      <td>192.00</td>
      <td>192.00</td>
      <td>192.00</td>
      <td>192.00</td>
      <td>-</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-11-28</td>
      <td>195.52</td>
      <td>195.52</td>
      <td>195.52</td>
      <td>195.52</td>
      <td>-</td>
      <td>0.0183</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>2021-02-04</td>
      <td>415.00</td>
      <td>415.00</td>
      <td>415.00</td>
      <td>415.00</td>
      <td>0.06K</td>
      <td>0.0234</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>2021-02-05</td>
      <td>408.00</td>
      <td>408.00</td>
      <td>408.00</td>
      <td>408.00</td>
      <td>0.12K</td>
      <td>-0.0169</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>2021-02-06</td>
      <td>415.00</td>
      <td>415.00</td>
      <td>415.00</td>
      <td>415.00</td>
      <td>0.04K</td>
      <td>0.0172</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2021-02-09</td>
      <td>417.00</td>
      <td>417.00</td>
      <td>417.00</td>
      <td>417.00</td>
      <td>0.09K</td>
      <td>0.0048</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>2021-02-10</td>
      <td>418.00</td>
      <td>418.00</td>
      <td>418.00</td>
      <td>418.00</td>
      <td>-</td>
      <td>0.0024</td>
    </tr>
  </tbody>
</table>
<p>1318 rows × 7 columns</p>
</div>



# Geometric Brownian Motion

This stochastic process has two components, a long-term trend and a short-term fluctuations. These two parts are also known as drift and diffusion respectively. I will provide very useful information for the understanding of this concept, since that was the base of the developing of this model. This [article](https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18) results very useful to code GBM.

## Drift

The long-term trend of the stock price is stored in this component. Dirft is rather a constant value and correnponds to the trend of the historical data. In order to reflect this in the calculations, mean and standard deviation of the return within the data interval of our data is needed. In other words, the mean and standard deviation of the daily growth is needed.

![](https://latex.codecogs.com/gif.latex?drift_{k}&space;=&space;\mu&space;-&space;\frac{1}{2}\sigma^{2})

## Diffusion

in order to calculate the short-term fluctuations, diffusion will do the job. This will help to compute different scenarios given by normal random values. These random numbers will multiply standard deviation of the historical return and will produce random day-by-day fluctuations on our simulations.

![](https://latex.codecogs.com/gif.latex?diffusion_{k}&space;=&space;\sigma&space;z_{k})

## Model coding

The formula that makes the prediction of the next day is an exponential growth expressed as follows:

![](https://latex.codecogs.com/gif.latex?S_{k}&space;=&space;S_{k-1}&space;*&space;e^{(drift_{k}&plus;diffusion_{k})})

If we substitute drift and diffusion in the previous formula, we get:

![](https://latex.codecogs.com/gif.latex?S_{k}&space;=&space;S_{k-1}&space;*&space;e^{(\mu&space;-&space;\frac{1}{2}&space;\sigma^{2}&space;&plus;&space;\sigma&space;z_{k})})

Below can be seen the model that allows the multiple predictions. With this, is possible to compute multiple scenarios that allows to analyse probabilities distributions along the predicted year.


```python
import numpy as np
so = np.array(data['ULT'].values)[-1]


def gbm(data,dt,T,N,t,esc):
    so = np.array(data['ULT'].values)[-1]
    ret = np.array(data['ULT'].values)
    mu = data['VAR'].mean()
    sigma = np.std(data['VAR'])
    b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, esc + 1)}
    W = {str(scen): b[str(scen)].cumsum() for scen in range(1, esc + 1)}

    drift = (mu - 0.5 * sigma**2) * t
    diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, esc + 1)}

    S1 = np.array([so * np.exp(drift + diffusion[str(scen)]) for scen in range(1, esc + 1)]) 
    S1 = np.hstack((np.array([[so] for scen in range(esc)]), S1))

    import datetime

    ini = data['FECHA'].max()
    fin = ini+datetime.timedelta(days=T)

    import matplotlib.pyplot as plt

    plt.figure(figsize = (20,10))
    plt.plot(data['FECHA'],data['ULT'])

    for i in range(esc):
        plt.title("Scenarios")
        plt.plot(pd.date_range(start = ini, end = fin, freq = str(dt)+'D'),S1[i,:])
        plt.ylabel('Monetary Unit')
        plt.xlabel('Prediction Days')

    plt.show()
    
    return S1
```

## Model test

Below can be seen that the model generates 3 scenarios.


```python
dt = 1#time step in the prediction
T = 365#duration of prediction
N = T/dt
t = np.arange(1, int(N)+1)
esc = 3#number of scenarios

S1 = gbm(data,dt,T,N,t,esc)
```


![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/3scenarios.png)


# Model prediction

A simulation of 1000 scenarios are generated to be able to analyse montly distributions, thus generating a large amount of data and have the most information available to make a decision in this example. 


```python
dt = 1
T = 365
N = T/dt
t = np.arange(1, int(N)+1)
esc = 1000

S1 = gbm(data,dt,T,N,t,esc)

```


![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/1000sce.png)



```python

```

## Month range and empty matrix

The next lines generate an empty matrix where data like monthly data, mean and standard deviation will be stored. Additionally, ranges of the months in the 365 days computed are stored.


```python
i = 0
M = np.arange(13)
md = np.zeros(shape=[1000,13])
rng = [[0,17],[18,48],[49,78],[79,109],[110,139],[140,170],[171,201],[202,231],[232,262],[263,293],[294,324],[325,355],[356,365]]
a = 0

for i in range(S1.shape[0]):
    a = 0
    for j in rng:
        md[i,a] = (S1[i][j]).mean()
        a += 1
```

## Storage and mean and standard deviation

The next months are stored in an array and the mean and standard deviation are also stored in the array.


```python
mdt = 4*md.transpose()#For this example a factor regarding money conversion is applied

feb,mar,abr,may,jun,jul,ago,sep,ocb,nov,dic,ene,feb2 = mdt[0],mdt[1],mdt[2],mdt[3],mdt[4],mdt[5],mdt[6],mdt[7],mdt[8],mdt[9],mdt[10],mdt[11],mdt[12]

mes = [feb,mar,abr,may,jun,jul,ago,sep,ocb,nov,dic,ene,feb2]

def std(mes):
    s = round(mes.std(),2)
    return s
    
def mean(mes):
    m = round(mes.mean(),2)
    return m

j = 0
for i in mes:
    i = mdt[j]
    j+=1

dtplot = [
            ['Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre','Enero 2022','Febrero 2022'],
            [feb,mar,abr,may,jun,jul,ago,sep,ocb,nov,dic,ene,feb2],
            [mean(feb),mean(mar),mean(abr),mean(may),mean(jun),mean(jul),mean(ago),mean(sep),mean(ocb),mean(nov),mean(dic),mean(ene),mean(feb2)],
            [std(feb),std(mar),std(abr),std(may),std(jun),std(jul),std(ago),std(sep),std(ocb),std(nov),std(dic),std(ene),std(feb2)]
           ]
```

# Visualisation of distributions

Information of scenarios computed are stored in the array called dtplot, then it is possible to show the monthly distribution of prices and mean and standard deviation.

The information provided, allows the client analise the chances of the prices of the goods she needs to make her products for her company.


```python
import matplotlib.pyplot as plt
k = 0
for k in np.arange(13):
    posx = (dtplot[1][k]).max()*1.05
    plt.figure(figsize=(4.5, 2.5))
    plt.text(posx, 100,'Promedio = ' + str(dtplot[2][k]),fontsize=12)
    plt.text(posx, 80,'Desv. Std. = ' + str(dtplot[3][k]),fontsize=12)
    plt.hist(dtplot[1][k],bins = 25)
    plt.title(dtplot[0][k])
    plt.show()
```


![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/feb1.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/mar.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/abr.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/may.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/jun.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/jul.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/ago.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/sep.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/oct.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/nov.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/dic.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/ene2.png)

![](https://github.com/highjoule/prices_prediction/blob/main/images/gbm2/feb22.png)


