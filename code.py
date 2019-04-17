"""
import nltk
#read some file in
f = open(“election2016.txt”)
#tokenize it using nltk
words = nltk.word_tokenize(f.read())
#pass words to nltk.FreqDist
freq = nltk.FreqDist(words)
#plot the top 25 words
freq.plot(25)
"""
import nltk
from textblob import TextBlob
import re
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 

from nltk.sentiment.vader import SentimentIntensityAnalyzer

with open('#Hillary.tab', 'r', encoding='utf-8') as myfile:
  data = myfile.read()
  
"""
with open('Clinton.txt', 'r') as myfile:
  data = myfile.read()
"""
import string
p = string.punctuation


d = string.digits


table = str.maketrans(p, len(p) * " ")
data= data.translate(table)

table = str.maketrans(d, len(d) * " ")
data=data.translate(table)

data = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data).split())



words = data.split()
stopwords = nltk.corpus.stopwords.words("english")
words = [w for w in words if w not in stopwords]


tweets = ' '.join(words)


querywords = tweets.split()
sw = ['b','xe','https','x','xa','twitter','d','sat','nov','co','com','I','status','web',
      'en','false','http','realdonaldtrump','download','i','true','hillary','a','makeamericagreatagain','one','people'
      ,'america','would','it','election','president','us','democrat','american','maga','anything']
resultwords  = [word for word in querywords if word.lower() not in sw]
tweets = ' '.join(resultwords)

"""
with open('stopwords1.txt', 'r') as myfile:
  s = myfile.read()
  
s = s.split(",")

querywords1 = tweets.split()
resultwords1  = [word for word in querywords if word.lower() not in s]
tweets = ' '.join(resultwords1)
"""
print("TEXT BLOB SENTIMENT ANALYSIS")
text = TextBlob(tweets)
print(text.sentiment)

print("NLTK Sentiment Analysis")
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(tweets)
print(ss)
score = ss['compound']
print(score)

content_hc=""


w = nltk.word_tokenize(tweets)
#pass words to nltk.FreqDist
freq = nltk.FreqDist(w)

freq.plot(25)

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(tweets) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


with open("r.png", "rb") as f_in:
    image_data=f_in.read()

with open("r.txt","w") as f_out:
    f_out.write(repr(image_data))
    
f = open("r.png","w")
f.write(content_hc)
f.close



img = Image.open("r1.png")
img = img.resize((2000,2000), Image.ANTIALIAS)
hcmask = np.array(img) 

wordcloud_hc = WordCloud(background_color="white", max_words=20000, mask=hcmask,stopwords=stopwords).generate(tweets)
wordcloud_hc.to_file("wc_hc.png")

plt.imshow(wordcloud_hc)
plt.axis("off")
plt.figure()
plt.imshow(hcmask, cmap=plt.cm.gray)
plt.axis("off")
plt.show() 