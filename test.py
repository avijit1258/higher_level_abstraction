from sklearn.feature_extraction.text import TfidfVectorizer
import re

txt1 = ['2 1 ', '2 1 ']

s = ' 2 1 10 '

print(re.findall(r'(\d+)', s) )

tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', token_pattern='\d+')
txt_fitted = tf.fit(txt1)
print(txt_fitted)
txt_transformed = txt_fitted.transform(txt1)
print(txt_transformed)

print('clear')