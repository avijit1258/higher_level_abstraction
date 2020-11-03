from sklearn.feature_extraction.text import TfidfVectorizer
import re

txt1 = ['2 1 ', '2 1 ']

s = ' 2 1 10 '

print(re.findall(r'(\d+)', s) )

#tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', token_pattern='\d+')
#txt2 = ['ausask  busas  husask  ', 'ausask  busas  gusask  ', 'ausask  cusask  eusask  ', 'ausask  cusask  dusask  gusask  ', 'ausask  cusask  dusask  fusask  ']
txt2 = ['His smile was not perfect', 'His smile was not not not not perfect', 'she not sang']
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', token_pattern='(?u)\\b\\w\\w+\\b')
txt_fitted = tf.fit(txt2)
print(txt_fitted)
txt_transformed = txt_fitted.transform(txt2)
print(txt_transformed)

print('clear')