import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#--------------------------------------------------------------------------------------------------------------------------

data_set= pd.read_csv('/Users/jiji/Desktop/Study/ML/Data.csv')       
   #read file and loads it into a pandas DataFrame

x= data_set.iloc[:, :-1].values    # select all values except last one
y= data_set.iloc[:,3].values       # select the fourth column

print(x)                           # independet variable
print(y)                           # dependent

#--------------------------------------------------------------------------------------------------------------------------


imputer = SimpleImputer(missing_values= np.nan , strategy='mean')  
   # fill in missing values /  specifies the placeholder for missing values / replaced by the average (mean) of that column

imputer= imputer.fit(x[:, 1:3])
   # calculates the statistics based on the data in the specified columns(1,2)

x[:, 1:3]= imputer.transform(x[:, 1:3])
   # transform(): replaces the missing values in the specified columns with the calculated mean
   # result is then assigned back to x[:, 1:3]

label_encoder_X = LabelEncoder()                   # OR--- x[:, 0] = LabelEncoder().fit_transform(x[:, 0])
x[:,0] = label_encoder_X.fit_transform(x[:,0])
   # categorical -> numerical values ; fits the encoder to the unique values in the 1st column

print('x after label encoding: \n', x)

#--------------------------------------------------------------------------------------------------------------------------


ct= ColumnTransformer(   # allows to apply different transformers (encoders, scalers) to specific columns
    transformers= [      # argument takes a list of transformers to be applied
        ('onehot',OneHotEncoder(), [0])      # each transformer is defined as a tuple with three elements
      # ('a name',transformer want to apply, list of column indices where the transformer should be applied)
      # OneHotEncoder: converts categorical labels into a binary (one-hot encoded) matrix
    ],remainder= 'passthrough'    #all other not being transformed columns should be passed through
)

x= ct.fit_transform(x)
   # first fits the OneHotEncoder to the first column, store it to x again
print('After OneHotEncoding:\n', x)

#--------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
   #  function is used to split a dataset into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  
   # random_state: ensures that the splitting is reproducible (will get the same train-test split each time run the code)


print('x_train:\n', x_train,)
print('x_test:\n', x_test)
print('y_train:\n', y_train)
print('y_test:\n', y_test)