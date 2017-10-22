# -*- coding: UTF-8 -*-
from xml.dom.minidom import parse
import xml.dom.minidom,sys
sys.stdout = open('data_train','w')
# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse("Restaurants_Train_v2.xml")
sentences = DOMTree.documentElement
sentence = sentences.getElementsByTagName("sentence")
count,count2=0,0
for sen in sentence:
   text=sen.getElementsByTagName('text')[0]
   aspectTerms=sen.getElementsByTagName('aspectTerms')
   if(len(aspectTerms)>0):
      data=text.childNodes[0].data
      data=data.replace('.',' .').replace(',',' ,').replace('n\'t',' n\'t').replace('\'s',' \'s').replace('perks',' perks ').replace('(',' ( ')
      data=data.replace(')',' ) ').replace('!',' !').replace('---',' -- ').replace('\'ll',' \'ll').replace(':',' :').replace('L .A .','L.A.')
      data=data.replace('\'re',' \'re').replace('day-','day').replace('\'ve',' \'ve').replace('$','$ ').replace('Ambiance-','Ambiance -')
      data=data.replace('\'doneness\'','\' doneness \'')
      print(data)
      count+=1
      aspectTerm=aspectTerms[0].getElementsByTagName('aspectTerm')
      count2+=len(aspectTerm)
      str=''
      for i in aspectTerm:
         str+=i.getAttribute('term')+' '
      print(str)

# print('total count is ',count)
# print('total target is ',count2)
