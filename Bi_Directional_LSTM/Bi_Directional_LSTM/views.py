from django.shortcuts import render  
from Bi_Directional_LSTM.forms import emailform


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM
from keras.layers import Embedding, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import load_model

  
def index(response):  
	if response.method == 'POST':
		data = emailform(response.POST) 
		if data.is_valid():
			content = data.cleaned_data['text']
			#content = 'I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.'
			print(content)
			model = load_model('Bi_Directional_LSTM/models/model.h5')
			new_complaint = [content]
			MAX_NB_WORDS = 50000
			MAX_SEQUENCE_LENGTH = 250

			df = pd.read_csv('Bi_Directional_LSTM/Bi LSTM Project/consumer_complaints.csv')

						#df data
			df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].fillna(" ")
			df = df.reset_index(drop=True)
			REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
			BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
			STOPWORDS = set(stopwords.words('english'))

			def clean_text(text):
				"""
				text: a string

				return: modified initial string
				"""

				text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
				text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
				text = text.replace('x', '')
				text = text.lower() # lowercase text
				#    text = re.sub(r'\W+', '', text)
				text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
				return text


			df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].apply(clean_text)
			df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].str.replace('\d+', '')


			#df data close




			# This is fixed.
			EMBEDDING_DIM = 100
			tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
			print(tokenizer)
			tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)
			seq = tokenizer.texts_to_sequences(new_complaint)
			print(seq)
			padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
			print(padded)
			pred = model.predict(padded)
			print(pred)
			labels = ['Mortgage', 'Debt collection', 'Credit reporting', 'Credit card', 'Bank account or service', 'Consumer Loan', 'Student loan', 'Payday loan', 'Money transfers', 'Prepaid card', 'Other financial service']
			print("\n\n", labels[np.argmax(pred)])

			return render(response, 'base.html', {'form': data, 'output':labels[np.argmax(pred)]})
		else:
			return render(response, 'base.html', {'form': data})

	else:
		data = emailform() 
		return render(response, 'base.html', {'form': data})


def index2(response):
	return render(response,'base2.html')

def index3(response):
	return render(response,'base3.html')