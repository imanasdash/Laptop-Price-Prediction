```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

```



```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```


```python
df = pd.read_csv('laptops.csv')
```


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Laptop</th>
      <th>Status</th>
      <th>Brand</th>
      <th>Model</th>
      <th>CPU</th>
      <th>RAM</th>
      <th>Storage</th>
      <th>Storage type</th>
      <th>GPU</th>
      <th>Screen</th>
      <th>Touch</th>
      <th>Final Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ASUS ExpertBook B1 B1502CBA-EJ0436X Intel Core...</td>
      <td>New</td>
      <td>Asus</td>
      <td>ExpertBook</td>
      <td>Intel Core i5</td>
      <td>8</td>
      <td>512</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>1009.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alurin Go Start Intel Celeron N4020/8GB/256GB ...</td>
      <td>New</td>
      <td>Alurin</td>
      <td>Go</td>
      <td>Intel Celeron</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>299.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ASUS ExpertBook B1 B1502CBA-EJ0424X Intel Core...</td>
      <td>New</td>
      <td>Asus</td>
      <td>ExpertBook</td>
      <td>Intel Core i3</td>
      <td>8</td>
      <td>256</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>789.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSI Katana GF66 12UC-082XES Intel Core i7-1270...</td>
      <td>New</td>
      <td>MSI</td>
      <td>Katana</td>
      <td>Intel Core i7</td>
      <td>16</td>
      <td>1000</td>
      <td>SSD</td>
      <td>RTX 3050</td>
      <td>15.6</td>
      <td>No</td>
      <td>1199.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HP 15S-FQ5085NS Intel Core i5-1235U/16GB/512GB...</td>
      <td>New</td>
      <td>HP</td>
      <td>15S</td>
      <td>Intel Core i5</td>
      <td>16</td>
      <td>512</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>15.6</td>
      <td>No</td>
      <td>669.01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Laptop</th>
      <th>Status</th>
      <th>Brand</th>
      <th>Model</th>
      <th>CPU</th>
      <th>RAM</th>
      <th>Storage</th>
      <th>Storage type</th>
      <th>GPU</th>
      <th>Screen</th>
      <th>Touch</th>
      <th>Final Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2155</th>
      <td>Razer Blade 17 FHD 360Hz Intel Core i7-11800H/...</td>
      <td>Refurbished</td>
      <td>Razer</td>
      <td>Blade</td>
      <td>Intel Core i7</td>
      <td>16</td>
      <td>1000</td>
      <td>SSD</td>
      <td>RTX 3060</td>
      <td>17.3</td>
      <td>No</td>
      <td>2699.99</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>Razer Blade 17 FHD 360Hz Intel Core i7-11800H/...</td>
      <td>Refurbished</td>
      <td>Razer</td>
      <td>Blade</td>
      <td>Intel Core i7</td>
      <td>16</td>
      <td>1000</td>
      <td>SSD</td>
      <td>RTX 3070</td>
      <td>17.3</td>
      <td>No</td>
      <td>2899.99</td>
    </tr>
    <tr>
      <th>2157</th>
      <td>Razer Blade 17 FHD 360Hz Intel Core i7-11800H/...</td>
      <td>Refurbished</td>
      <td>Razer</td>
      <td>Blade</td>
      <td>Intel Core i7</td>
      <td>32</td>
      <td>1000</td>
      <td>SSD</td>
      <td>RTX 3080</td>
      <td>17.3</td>
      <td>No</td>
      <td>3399.99</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>Razer Book 13 Intel Evo Core i7-1165G7/16GB/1T...</td>
      <td>Refurbished</td>
      <td>Razer</td>
      <td>Book</td>
      <td>Intel Evo Core i7</td>
      <td>16</td>
      <td>1000</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>13.4</td>
      <td>Yes</td>
      <td>1899.99</td>
    </tr>
    <tr>
      <th>2159</th>
      <td>Razer Book FHD+ Intel Evo Core i7-1165G7/16GB/...</td>
      <td>Refurbished</td>
      <td>Razer</td>
      <td>Book</td>
      <td>Intel Evo Core i7</td>
      <td>16</td>
      <td>256</td>
      <td>SSD</td>
      <td>NaN</td>
      <td>13.4</td>
      <td>Yes</td>
      <td>1699.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (2160, 12)




```python
df.size
```




    25920




```python
df.columns

```




    Index(['Laptop', 'Status', 'Brand', 'Model', 'CPU', 'RAM', 'Storage',
           'Storage type', 'GPU', 'Screen', 'Touch', 'Final Price'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2160 entries, 0 to 2159
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Laptop        2160 non-null   object 
     1   Status        2160 non-null   object 
     2   Brand         2160 non-null   object 
     3   Model         2160 non-null   object 
     4   CPU           2160 non-null   object 
     5   RAM           2160 non-null   int64  
     6   Storage       2160 non-null   int64  
     7   Storage type  2118 non-null   object 
     8   GPU           789 non-null    object 
     9   Screen        2156 non-null   float64
     10  Touch         2160 non-null   object 
     11  Final Price   2160 non-null   float64
    dtypes: float64(2), int64(2), object(8)
    memory usage: 202.6+ KB



```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RAM</th>
      <th>Storage</th>
      <th>Screen</th>
      <th>Final Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2160.000000</td>
      <td>2160.000000</td>
      <td>2156.000000</td>
      <td>2160.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15.413889</td>
      <td>596.294444</td>
      <td>15.168112</td>
      <td>1312.638509</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.867815</td>
      <td>361.220506</td>
      <td>1.203329</td>
      <td>911.475417</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>10.100000</td>
      <td>201.050000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.000000</td>
      <td>256.000000</td>
      <td>14.000000</td>
      <td>661.082500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.000000</td>
      <td>512.000000</td>
      <td>15.600000</td>
      <td>1031.945000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.000000</td>
      <td>1000.000000</td>
      <td>15.600000</td>
      <td>1708.970000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>128.000000</td>
      <td>4000.000000</td>
      <td>18.000000</td>
      <td>7150.470000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    Laptop             0
    Status             0
    Brand              0
    Model              0
    CPU                0
    RAM                0
    Storage            0
    Storage type      42
    GPU             1371
    Screen             4
    Touch              0
    Final Price        0
    dtype: int64




```python
df.drop(['GPU'] , axis=1, inplace=True)
```


```python
df.dropna(inplace=True)
```


```python
df.duplicated().value_counts()
```




    False    2114
    Name: count, dtype: int64




```python
df.nunique()
```




    Laptop          2114
    Status             2
    Brand             27
    Model            119
    CPU               27
    RAM                9
    Storage           11
    Storage type       2
    Screen            28
    Touch              2
    Final Price     1409
    dtype: int64




```python
plt.figure(figsize=(8,6))
plt.title('Distribution of Storage Types')
sizes = df['Storage type'].value_counts()
plt.pie(sizes, labels=sizes.index, autopct='%1.2f%%')
plt.show()
```


 ![Unknown-1](https://github.com/user-attachments/assets/f88ab322-881f-4c67-9912-076fd779c113)
   

    



```python
plt.figure(figsize=(8,6))
plt.title('Distribution of Status')
sizes = df['Status'].value_counts()
plt.pie(sizes, labels = sizes.index, autopct = '%1.1f%%')
plt.show()
```


    
![Unknown-2](https://github.com/user-attachments/assets/616a5ea9-bc0e-4d87-805e-fb0c410b1b7f)

    



```python
plt.figure(figsize=(8,6))
plt.title('Distribution Of Touch')
sizes = df['Touch'].value_counts()
plt.pie(sizes, labels=sizes.index, autopct = '%1.1f%%')
plt.show()
```


![Unknown-3](https://github.com/user-attachments/assets/96055cc1-4db6-4136-8687-40c1ce71c100)

    



```python
plt.figure(figsize = (6,6))
plt.title('Count of Brand of Laptops')
counts = df['Brand'].value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel('Brand')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()
```

![Unknown-4](https://github.com/user-attachments/assets/1c3a81e3-3316-4be4-b571-d29bac6c846d)


    



```python
plt.figure(figsize=(6,6))
plt.title('Count of CPU')
counts = df['CPU'].value_counts()
sns.barplot(x=counts.index,y=counts.values)
plt.xlabel('CPU')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```


![Unknown-5](https://github.com/user-attachments/assets/c5d417a3-9e49-4f65-b012-31d25d245ec7)


    



```python
plt.figure(figsize=(23,10))
plt.title('Count of Models')
counts = df['Model'].value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel('Models')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()
```


![Unknown](https://github.com/user-attachments/assets/d4ae9c97-8cd0-45e0-b30e-9ca728950c9d)
 

    



```python
plt.figure(figsize=(8,6))
plt.title('Distribution of RAM')
plt.hist(df['RAM'],bins=10)
plt.xlabel('RAM')
plt.ylabel('Frequency')
plt.show()
```


    
![Unknown-6](https://github.com/user-attachments/assets/f7bffbea-ee73-405b-8913-ea625411f495)

    



```python
plt.figure(figsize=(8,6))
plt.title('Distribution of Storage')
plt.hist(df['Storage'],bins=10)
plt.xlabel('Storage')
plt.ylabel('Frequency')
plt.show()

```


![Unknown-7](https://github.com/user-attachments/assets/9c63667f-8608-4505-a503-461b7b5b6e62)


    



```python
plt.figure(figsize=(8,6))
plt.hist(df['Screen'],bins=10)
plt.xlabel('Screen')
plt.ylabel('Frequency')
plt.title('Distribution of Screen')
plt.show()
```

![Unknown-8](https://github.com/user-attachments/assets/fe7a6ceb-fdf0-4c36-86c7-8163ae6704c2)

    

    


### Feature Engineering


```python
#label encoding
label_encoder = LabelEncoder()
df['Status_encoded'] = label_encoder.fit_transform(df['Status'])
df['Storage_encoded'] = label_encoder.fit_transform(df['Storage type'])
df['Touch_encoded'] = label_encoder.fit_transform(df['Touch'])
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2114 entries, 0 to 2159
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Laptop           2114 non-null   object 
     1   Status           2114 non-null   object 
     2   Brand            2114 non-null   object 
     3   Model            2114 non-null   object 
     4   CPU              2114 non-null   object 
     5   RAM              2114 non-null   int64  
     6   Storage          2114 non-null   int64  
     7   Storage type     2114 non-null   object 
     8   Screen           2114 non-null   float64
     9   Touch            2114 non-null   object 
     10  Final Price      2114 non-null   float64
     11  Status_encoded   2114 non-null   int64  
     12  Storage_encoded  2114 non-null   int64  
     13  Touch_encoded    2114 non-null   int64  
    dtypes: float64(2), int64(5), object(7)
    memory usage: 247.7+ KB



```python
df.drop(['Laptop','Status','Brand','Model','CPU','Storage type','Touch'],axis=1,inplace=True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2114 entries, 0 to 2159
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   RAM              2114 non-null   int64  
     1   Storage          2114 non-null   int64  
     2   Screen           2114 non-null   float64
     3   Final Price      2114 non-null   float64
     4   Status_encoded   2114 non-null   int64  
     5   Storage_encoded  2114 non-null   int64  
     6   Touch_encoded    2114 non-null   int64  
    dtypes: float64(2), int64(5)
    memory usage: 132.1 KB



```python
corr_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',square=True)
plt.title('Correlation Heatmap')
plt.show()
```


![Unknown-9](https://github.com/user-attachments/assets/9176eccc-4264-4f64-a764-18e4d13519de)


    


## Data Modeling


```python
X = df[['RAM','Storage','Screen','Status_encoded','Storage_encoded','Touch_encoded']]
y = df['Final Price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
```


```python
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red',linestyle='dotted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
```


    
![Unknown-10](https://github.com/user-attachments/assets/61ecf877-6dc8-444a-a10c-b392bdf2ff21)

    


### Model Evaluation 


```python
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)
print('R-squared (R2 Score):', r2)
```

    Mean Squared Error (MSE): 323695.691237309
    Root Mean Squared Error (RMSE): 568.9426080346848
    Mean Absolute Error (MAE): 401.32732097295855
    R-squared (R2 Score): 0.5832382262482508



```python
coefficients = model.coef_
intercept = model.intercept_ 
print('Coefficients:', coefficients) 
print('Intercept:', intercept)
```

    Coefficients: [  42.22482889    0.90614025    8.57482185 -211.39911504 -127.34749912
      455.95399793]
    Intercept: 13.878065598763897



```python
new_laptop = np.array([[8, 256, 15.6, 0, 1, 1]])
predicted_price = model.predict(new_laptop) 
print('Predicted Price:', predicted_price)
```

    Predicted Price: [1046.0223202]

```
