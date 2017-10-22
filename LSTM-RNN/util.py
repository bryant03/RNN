import numpy as np
import re
def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) ):
            yield index[i * batch_size:(i + 1) * batch_size]

# 将每个单词进行标号，这样在存储的时候就不需要存储整个句子，只存储标号就好
def load_word_id_mapping(word_id_file,encoding='utf8'):
    print('word_id_file',word_id_file)
    count=0
    word_to_id=dict()
    for line in open(word_id_file,'rb'):
        # count+=1
        # if count>10:
        #     break
        line=line.decode(encoding,'ignore').lower().split()
        word_to_id[line[0]]=int(line[1])
        # print(line[0],line[1])
    print ('\nload word-id mapping done!\n')
    return word_to_id


# 将word2vec导入
def load_w2v(w2v_path,embedding_dim):
    fp=open(w2v_path)
    w2v=[]
    word_dict=dict()
    w2v.append([0.]*embedding_dim)
    cnt=0
    for line in fp:
        cnt+=1
        line=line.split ()
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]]=cnt
    w2v=np.asarray(w2v,dtype=np.float32)
    return word_dict,w2v

def load_word_embedding(word_id_file,w2v_path,embedding_dim):
    word_to_id=load_word_id_mapping(word_id_file)
    word_dict,w2v=load_w2v(w2v_path,embedding_dim)
    cnt=len(w2v)
    for k in word_to_id.keys():
        if k not in word_dict:
            word_dict[k] = cnt
            w2v = np.row_stack((w2v, np.random.uniform(-0.01, 0.01, (embedding_dim,))))
            cnt += 1
    # print (len(word_dict), len(w2v)) 
    return word_dict, w2v
def change_y_to_onehot(y):
    print(np.shape(y))
    onehot=[]
    for i in y:
        onehot_line=[]
        for j in i:
            tmp=[0,0]
            tmp[j]=1
            onehot_line.append(tmp)
        onehot.append(onehot_line)
    # return onehot
    return np.asarray(onehot, dtype=np.float32)

def load_inputs(input_file,word_id_file,sentence_len):
    encoding='utf8'
    word_to_id=word_id_file
    x,y_t,sen_len=[],[],[]
    lines=open(input_file,'rb').readlines()
    maxx=0
    for i in range(0,len(lines),2):
        words=lines[i].decode(encoding).lower().split()
        aspects=lines[i+1].decode(encoding).lower().split()
        ids=[]
        i=0
        y=[]
        aspects_len=len(aspects)
        # new_words=[]
        # for word in words:
        #     ll=len(word)
        #     if ll>3 and word[ll-3:ll]=='n\'t':
        #         new_words.append(word[:ll-3])
        #         new_words.append('n\'t')
        #     else:
        #         new_words.append(word)
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
                if (i<aspects_len and word == aspects[i]):
                    y.append(1)
                    i+=1
                else :
                    y.append(0)
            else: 
                ids.append(0)
        # if maxx<len(ids):
        #     maxx=len(ids)
        #     print(maxx)
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids)))
        y_t.append(y+[0] * (sentence_len - len(y)))
    # print('maxx = ',maxx)
    y_t=change_y_to_onehot(y_t)
    return np.asarray(x,dtype=np.float32),np.asarray(sen_len), np.asarray(y_t,dtype= np.float32)

if __name__ == '__main__':
    encoding='utf8'
    y=[[1,0,1],
    [0,0,0]]
    print(change_y_to_onehot(y))
    # sentence_len=30
    # x=[]
    # y_t=[]
    # lines=open('data/restaurant/data_test','rb').readlines()
    # for i in range(0,len(lines),2):
    #     words=lines[i].decode(encoding).lower().split()
    #     aspects=lines[i+1].decode(encoding).lower().split()
    #     ids=[]
    #     i=0
    #     y=[]
    #     aspects_len=len(aspects)
    #     for word in words:
    #         ids.append(1)
    #         if (i<aspects_len and word == aspects[i]):
    #             y.append(1)
    #             i+=1
    #         else :
    #             y.append(0)

    #     # sen_len.append(len(ids))
    #     x.append(ids + [0] * (sentence_len - len(ids)))
    #     y_t.append(y+[0] * (sentence_len - len(ids)))
    #     print(ids + [0] * (sentence_len - len(ids)))
    #     print(y+[0] * (sentence_len - len(ids)))
    # print(np.shape(np.asarray(x,dtype=np.float32)))
    # print(np.shape(np.asarray(y_t,dtype=np.float32)))