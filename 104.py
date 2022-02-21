# 匯入套件
import jieba
import jieba.posseg
import jieba.analyse
from collections import Counter
from pprint import pprint
from wordcloud import WordCloud
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 載自己的判決檔案
df = pd.read_csv('./onlyBillText.csv',header=None,sep=',', encoding = 'utf-8')

# 0 id 1 有無前科(rec) 2 場域(place) 3 和解(compr) 4 內容(Xword) 5 學歷(edu) 
# 6 智識(mind) 7 經濟(fin) 8 扶養(support) 9 態度(att) 10 坦承(confess) 
# 11 接續犯(sequel) 12 罪刑(law) 13 金額(label)

# 目標: 用 dict 一行一行 column insert 上去
# 先試試 insert id

# 建立空dict
dfdict={}

# 看遺漏值是在哪一row
df[df.isnull().values==True]

# 10156 的中間欄位沒有值
# drop index 10156的遺漏值
df_dropna = df.drop(df.index[[10156]])
# reset index
df_dropna.reset_index(inplace=True, drop=True)

# 取得 第一column id 的list
id1=list(df_dropna[0])
#print(id1)

# 讀資料取得第二column的 有無前科(rec)
# 讀取 record.txt
rec = open('./featuretocol/record.txt','r',encoding='utf-8')
lines = rec.readlines()
rec_word = [] 
for line in lines:
    line = line.replace('\n','')
    rec_word.append(line)
    
# 讀取第三column的 場域 
plc = open('./featuretocol/place.txt','r',encoding='utf-8')
lines = plc.readlines()
plc_word = [] 
for line in lines:
    line = line.replace('\n','')
    plc_word.append(line)

# 讀取第四column 的 和解意願 compromise
cmpr = open('./featuretocol/compromise.txt','r',encoding='utf-8')
lines = cmpr.readlines()
cmpr_word = [] 
for line in lines:
    line = line.replace('\n','')
    cmpr_word.append(line)

# 讀取第五column的 手段內容 Xword
xwd = open('./featuretocol/Xword.txt','r',encoding='utf-8')
lines = xwd.readlines()
xwd_word = [] 
for line in lines:
    line = line.replace('\n','')
    xwd_word.append(line)
    
# 讀取第六column的 學歷 edu
edu = open('./featuretocol/education.txt','r',encoding='utf-8')
lines = edu.readlines()
edu_word = [] 
for line in lines:
    line = line.replace('\n','')
    edu_word.append(line)
    
# 讀取第7column的 智識 mind
mind = open('./featuretocol/mind.txt','r',encoding='utf-8')
lines = mind.readlines()
mind_word = [] 
for line in lines:
    line = line.replace('\n','')
    mind_word.append(line)

# 讀取第八column的 fin 經濟
fin = open('./featuretocol/financial.txt','r',encoding='utf-8')
lines = fin.readlines()
fin_word = [] 
for line in lines:
    line = line.replace('\n','')
    fin_word.append(line)
    
# 讀取第九column的 sup 扶養
sup = open('./featuretocol/support.txt','r',encoding='utf-8')
lines = sup.readlines()
sup_word = [] 
for line in lines:
    line = line.replace('\n','')
    sup_word.append(line)
    
# 讀取第10 column的 態度 att
att = open('./featuretocol/attitude.txt','r',encoding='utf-8')
lines = att.readlines()
att_word = [] 
for line in lines:
    line = line.replace('\n','')
    att_word.append(line)
    
# 讀取 第11 column的 confess 坦承
cfs = open('./featuretocol/confess.txt','r',encoding='utf-8')
lines = cfs.readlines()
cfs_word = [] 
for line in lines:
    line = line.replace('\n','')
    cfs_word.append(line)
    
# 讀取 第12 column的 sequel 接續
seql = open('./featuretocol/sequel.txt','r',encoding='utf-8')
lines = seql.readlines()
seql_word = [] 
for line in lines:
    line = line.replace('\n','')
    seql_word.append(line)
    
# 讀取締13 column 的 罪刑 law
law = open('./featuretocol/law.txt','r',encoding='utf-8')
lines = law.readlines()
law_word = [] 
for line in lines:
    line = line.replace('\n','')
    law_word.append(line)

# 讀取第14 column 的 金額 label
lab14=list(df_dropna[2])
#print(lab14)
    
    
# 讀取總特徵辭庫 作為結疤的分詞

jieba.load_userdict('./featuretocol/totalfeature.txt')

# 0 id 1 有無前科(rec) 2 場域(place) 3 和解(compr) 4 內容(Xword) 5 學歷(edu) 
# 6 智識(mind) 7 經濟(fin) 8 扶養(support) 9 態度(att) 10 坦承(confess) 
# 11 接續犯(sequel) 12 罪刑(law) 13 金額(label)

# 1 id 2 rec 3 plc 4 cmpr 5 xwd 6 edu 7 mind 8 fin 9 sup 10 att 11 cfs 
# 12 seql 13 law 14 lab 

# 建立個欄位的空list
listofrec=[]
listofplc=[]
listofcmpr=[]
listofxwd=[]
listofedu=[]
listofmind=[]
listoffin=[]
listofsup=[]
listofatt=[]
listofcfs=[]
listofseql=[]
listoflaw=[]



for i in range(len(df_dropna[0])):
    totalwords = df_dropna[1][i]
    # 判決書 初次以總特徵辭庫做分詞
    seg = jieba.lcut(totalwords, cut_all=False)
    
    # 以rec特徵辭庫去看 seg裡面 重複的詞是什麼
    onerec = list(set(filter(lambda a : a in rec_word, seg)))
    
    # 看每一判決書有沒有特徵詞, 一個一個append上去
    # 如果是空集合, 轉換成空字串
    if len(onerec) > 0:
        listofrec.append(onerec)
    else:
        listofrec.append(str(onerec).replace('[]',''))
    
    # 對比 place 特徵詞
    oneplc = list(set(filter(lambda a : a in plc_word, seg)))
    
    if len(oneplc) > 0:
        listofplc.append(oneplc)
    else:
        listofplc.append(str(oneplc).replace('[]',''))
        
    # 對比 compromise 特徵詞
    onecmpr = list(set(filter(lambda a : a in cmpr_word, seg)))
    
    if len(onecmpr) > 0:
        listofcmpr.append(onecmpr)
    else:
        listofcmpr.append(str(onecmpr).replace('[]',''))
    
    # 對比 xword 特徵詞
    onexwd = list(set(filter(lambda a : a in xwd_word, seg)))
    if len(onexwd) > 0:
        listofxwd.append(onexwd)
    else:
        listofxwd.append(str(onexwd).replace('[]',''))
    
    # 對比 education 特徵詞
    oneedu = list(set(filter(lambda a : a in edu_word, seg)))
    if len(oneedu) > 0:
        listofedu.append(oneedu)
    else:
        listofedu.append(str(oneedu).replace('[]',''))
    
    # 對比 mind 特徵詞
    onemind = list(set(filter(lambda a : a in mind_word, seg)))
    if len(onemind) > 0:
        listofmind.append(onemind)
    else:
        listofmind.append(str(onemind).replace('[]',''))
        
    # 對比 financial 特徵詞
    onefin = list(set(filter(lambda a : a in fin_word, seg)))
    if len(onefin) > 0:
        listoffin.append(onefin)
    else:
        listoffin.append(str(onefin).replace('[]',''))
        
    # 對比 support 特徵詞
    onesup = list(set(filter(lambda a : a in sup_word, seg)))
    if len(onesup) > 0:
        listofsup.append(onesup)
    else:
        listofsup.append(str(onesup).replace('[]',''))
    
    # 對比 attitude 特徵詞
    oneatt = list(set(filter(lambda a : a in att_word, seg)))
    if len(oneatt) > 0:
        listofatt.append(oneatt)
    else:
        listofatt.append(str(oneatt).replace('[]',''))
    
    # 對比 confess 特徵詞
    onecfs = list(set(filter(lambda a : a in cfs_word, seg)))
    if len(onecfs) > 0:
        listofcfs.append(onecfs)
    else:
        listofcfs.append(str(onecfs).replace('[]',''))
        
    # 對比 sequel 特徵詞
    oneseql = list(set(filter(lambda a : a in seql_word, seg)))
    if len(oneseql) > 0:
        listofseql.append(oneseql)
    else:
        listofseql.append(str(oneseql).replace('[]',''))
    
    # 對比 law 特徵詞
    onelaw = list(set(filter(lambda a : a in law_word, seg)))
    if len(onelaw) > 0:
        listoflaw.append(onelaw)
    else:
        listoflaw.append(str(onelaw).replace('[]',''))
    
# 1 id 2 rec 3 plc 4 cmpr 5 xwd 6 edu 7 mind 8 fin 9 sup 10 att 11 cfs 
# 12 seql 13 law 14 lab 


# 最後建立 dict, 一一補上column
dfdict = {
    "id":id1,
    "record":listofrec,
    "place":listofplc,
    "compromise":listofcmpr,
    "xword":listofxwd,
    "education":listofedu,
    "mind":listofmind,
    "financial":listoffin,
    "support":listofsup,
    "attitude":listofatt,
    "confess":listofcfs,
    "sequel":listofseql,
    "law":listoflaw,
    "label":lab14
    }

# 最終完成的 dataframe
dffinal = pd.DataFrame(dfdict)

dffinal.to_csv("./featuretocol/feattocol.csv", header= True, index= True, encoding='UTF-8')


print(dffinal.head(5))


 
