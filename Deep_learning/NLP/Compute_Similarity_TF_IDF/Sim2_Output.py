#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
from simhash import Simhash


# In[2]:


class Similarity:
    # TF计算
    def tf_similarity(self,str1,str2):
        def add_space(s):
            return ' '.join(list(s))
        
        str1, str2 = add_space(str1),add_space(str2)
        cv = CountVectorizer(tokenizer=lambda s:s.split())
        corpus = [str1,str2]
        vectors = cv.fit_transform(corpus).toarray()
        
        return np.dot(vectors[0],vectors[1]) /(norm(vectors[0]) * norm(vectors[1]))
    
    
    #TF-IDF
    def tfidf_similarty(self,str1,str2):
        def add_space(s):
            return ' '.join(list(s))
        
        str1, str2 = add_space(str1),add_space(str2)
        cv = TfidfVectorizer(tokenizer=lambda s:s.split())
        corpus = [str1,str2]
        vectors = cv.fit_transform(corpus).toarray()
        
        return np.dot(vectors[0],vectors[1]) /(norm(vectors[0]) * norm(vectors[1]))
    
    
    # simhash算法
    def simhash_similarity(self,str1,str2):
        aa_simhash = Simhash(str1)
        bb_simhash = Simhash(str2)
        max_hashbit = max(len(bin(aa_simhash.value)),(len(bin(bb_simhash.value))))
        
        distince = aa_simhash.distance(bb_simhash)
        similar = 1 - distince / max_hashbit
        return similar

    # 输出最终相似度
    def get_similarity(self,str1,str2):
        similarity1 = self.tf_similarity(str1,str2)
        similarity2 = self.tfidf_similarty(str1,str2)
        similarity3 = self.simhash_similarity(str1,str2)
        
        similarity_final = (1.5*similarity1+1.3*similarity2+0.2*similarity3)/3
        # 四位小数 99.88%
        return "%.4f"%similarity_final


# In[3]:


model = Similarity()


# In[ ]:





# ## 接口输出

# In[4]:


import logging
import logging.handlers
from gevent import pywsgi
from flask import request, Flask, Response

logging.basicConfig(level=logging.DEBUG)
filehandler = logging.handlers.TimedRotatingFileHandler("Logs/Recflask.log", "D", 1, 0)
logging.getLogger().addHandler(filehandler)


# In[5]:


app = Flask(__name__)

@app.route('/heartbeat/', methods=['GET', 'HEAD'])
def checkHeartBeat():
    return '1'

@app.route('/similarity/', methods=['POST'])
def answers1():
    try:
        str1 = request.form.get('str1')
        str2 = request.form.get('str2')
        data = model.get_similarity(str1,str2)
        result = {'code': '200', 'data': data,'msg': "success"}
    except Exception as e:
        result = {'code': '101', 'data': [str(e)],'msg': "error"}
        logging.error('Error: %s'%(str(e)))
    return result


# In[ ]:


if __name__ == '__main__':
    #开发环境
    #app.run(host='0.0.0.0', port=11002, debug=False) #,use_reloader=False)
    server = pywsgi.WSGIServer(('0.0.0.0', 11002), app)
    server.serve_forever()


# In[ ]:




