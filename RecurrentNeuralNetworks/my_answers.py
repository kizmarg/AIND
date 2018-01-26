import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # length of series
    series_length = len(series)
    
    for index in range(0, series_length):
        if (series_length-index-window_size)>0:
            X.append(series[index:index+window_size])
            y.append(series[index+window_size])
            
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()    
    
    # layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(5, input_shape=(window_size, 1)))   
    
    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1)) # one unit (Question : I think it should be at least two units???)
    
    return model
    
  
import re  

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
 # Because of this, the effect of line.replace(...) is just to create a new string, rather than changing the old one. 
    punctuation.append(' ')
    
    regex_rule = '[^a-zA-Z'+ re.escape(''.join(punctuation)) +']'
    regex = re.compile(regex_rule)
    #First parameter is the replacement, second parameter is your input string
    text = regex.sub('', text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    
    index = 0 
    while (len(text)-index-window_size)>0:
        inputs.append(text[index:index+window_size])
        outputs.append(text[index+window_size])
        index = index + step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    # layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    # layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    model.add(Dense(num_chars))
    # layer 3 should be a softmax activation ( since we are solving a multiclass classification)
    model.add(Activation('softmax'))
    # Use the categorical_crossentropy loss
    
    return model
