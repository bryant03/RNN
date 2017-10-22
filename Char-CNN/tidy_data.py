import numpy as np
def load_charId(embedding_dim=68,Debug=False):
    chardic=['a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9',
    ',',';','.','!','?',':','\'','\"','/','\\','|','_',
    '@','#','$','%','^','&','*','+','-','=','<','>','(',
    ')','[',']','{','}','`','~']
    char_dic=dict()
    w2v=np.array([[0.]*embedding_dim]*(embedding_dim+1))
    for i in range(embedding_dim):
        w2v[i+1][i]=1.
        char_dic[chardic[i]]=i+1
    return char_dic,w2v
def load_inputs_document_forcharCNN(input_file, char_to_id, max_doc_len=1014, n_class=5, encoding='utf8'):
    x, y, doc_len = [], [], []
    forloop=0
    for line in open(input_file,'rb'):
        # print(line.lower())
        # print(type(line))
        line=line.decode('utf8')
        # print(type(line))
        line=line.lower().split('\t\t')
        # print(line)
        # print(type(line))
        y.append(int(line[0]))
        # print(line[1])
        # t_sen_len=[0]*max_doc_len
        t_x=np.zeros((max_doc_len),dtype=np.int)
        doc=' '.join(line[1:])
        # print(type(line[1:]))
        sentences=doc.split('<sssss>')
        sentences=' '.join(sentences)
        i=0
        # print('sent is {}'.format(sentences))
        for ch in sentences:
            if ch in char_to_id:
                t_x[i]=char_to_id[ch]
            i+=1
            if i>=max_doc_len:
                break
        doc_len.append(i)
        # print(t_x)
        # print('shape t_x is {}'.format(np.shape(t_x)))
        x.append(t_x)
        # forloop+=1
        # if(forloop>=3):
        #     break
    print('shape y is {}'.format(np.shape(y)))
    print('shape n_class is {}'.format(np.shape(n_class)))
    print('shape x is {}'.format(np.shape(x)))
    y= change_y_to_onehot(y,n_class)
    print('prepare done!')
    print(np.shape(np.asarray(x)))
    return np.asarray(x),np.asarray(y),np.asarray(doc_len)
def change_y_to_onehot(y,n_class=5):
    onehot=[]
    for i in y:
        tmp=[0]*n_class
        tmp[i-1]=1
        onehot.append(tmp)
    return np.asarray(onehot,dtype=np.int32)
def batch_index(length,bath_size,n_iter=100):
    print("length is {}".format(length))
    index=list(range(length))
    for j in range(n_iter):
        np.random.shuffle(index)
        for i in range(int(length/bath_size)):
            yield index[i*bath_size:(i+1)*bath_size]
            
if __name__ =='__main__':
    char_id_mapping,c2v=load_charId()
    # load_inputs_document_forcharCNN('data/min0.1_yelp-2013-train.txt.ss',char_id_mapping,1014)
