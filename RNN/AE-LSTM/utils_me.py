import numpy as np 


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]

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


def change_y_to_onehot(y):
    from collections import Counter
    print (Counter(y))
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

def load_w2v(w2v_file,embedding_dim,is_skip=False):

    fp=open(w2v_file)
    if is_skip:
        fp.readline()
    w2v=[]
    word_dict=dict()
    w2v.append([0.]*embedding_dim)
    cnt=0
    for line in fp:
        cnt+=1
        line=line.split()
        # print(len(line))  ---301
        if len(line)!=embedding_dim+1 :
            print('a bad word embedding: {}'.format(line[0]))
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print ('w2v shape',np.shape(w2v)) #3774 300
    print ('word_dict shape',np.shape(word_dict)) #3774 1
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print (word_dict['$t$'], len(w2v))
    return word_dict, w2v

def load_word_embedding(word_id_file,w2v_file,embedding_dim,is_skip=False):
    word_to_id=load_word_id_mapping(word_id_file)
    word_dict,w2v=load_w2v(w2v_file,embedding_dim,is_skip)
    cnt=len(w2v)
    print("len w2v:",cnt)
    print("len word_dict:",len(word_dict))
    for k in word_to_id.keys():
        if k not in word_dict:
            word_dict[k] = cnt
            w2v = np.row_stack((w2v, np.random.uniform(-0.01, 0.01, (embedding_dim,))))
            cnt += 1
    print (len(word_dict), len(w2v)) #3774 3775
    return word_dict, w2v

def load_aspect2id(input_file,word_id_mapping,w2v,embedding_dim):
    aspect2id = dict()
    a2v = list()
    a2v.append([0.] * embedding_dim)
    cnt = 0
    for line in open(input_file):
        line = line.lower().split()
        cnt += 1
        aspect2id[' '.join(line[:-1])] = cnt
        tmp = []
        for word in line:
            if word in word_id_mapping:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp:
            a2v.append(np.sum(tmp, axis=0) / len(tmp))
        else:
            a2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
    print (len(aspect2id), len(a2v))
    print ("shape aspect2id: ",len(aspect2id))

    return aspect2id, np.asarray(a2v, dtype=np.float32)    




def load(input_file,word_id_mapping,w2v,embedding_dim):

    aspect2id=dict()
    a2v=list()
    a2v.append([0.*embedding_dim])
    cnt=0
    for line in open(input_file):
        print(line)
        line=line.lower().split()
        cnt+=1
        aspect2id[' '.join(line[:-1])] = cnt
        tmp=[]
        for word in line:
            if word in word_id_mapping:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp:
            print(len(tmp))
            print(np.sum(tmp,axis=0)/len(tmp))
            a2v.append(np.sum(tmp,axis=0)/len(tmp))
            break



def load_inputs_twitter_at(input_file, word_id_file, aspect_id_file, sentence_len, type_='', encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print ('load word-to-id done!')
    if type(aspect_id_file) is str:
        aspect_to_id = load_aspect2id(aspect_id_file)
    else:
        aspect_to_id = aspect_id_file
    print ('load aspect-to-id done!')

    x, y, sen_len = [], [], []
    aspect_words = []
    lines = open(input_file,'rb').readlines()
    for i in range(0, len(lines), 3):
        aspect_word = ' '.join(lines[i + 1].decode(encoding).lower().split())
        aspect_words.append(aspect_to_id.get(aspect_word, 0))

        y.append(lines[i + 2].split()[0])

        words = lines[i].decode(encoding).lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids)))
    cnt = 0
    for item in aspect_words:
        if item > 0:
            cnt += 1
    print ('cnt=', cnt)
    y = change_y_to_onehot(y)
    for item in x:
        if len(item) != sentence_len:
            print ('aaaaa=', len(item))
    x = np.asarray(x, dtype=np.int32)

    return x, np.asarray(sen_len), np.asarray(aspect_words), np.asarray(y)
if __name__ == '__main__':
    word_id_file_path='data/restaurant/word_id_new.txt'
    embedding_file_path='data/restaurant/rest_2014_word_embedding_300_new.txt'
    aspect_id_file_path='data/restaurant/aspect_id_new.txt'
    # word_id_mapping,w2v=load_word_embedding('data/restaurant/word_id_new.txt','data/restaurant/rest_2014_word_embedding_300_new.txt',300)
    # load_aspect2id('data/restaurant/aspect_id_new.txt',word_id_mapping,w2v,300)

    word_id_mapping,w2v=load_word_embedding(word_id_file_path,embedding_file_path, 300)
    # word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding')
    aspect_id_mapping, aspect_embed = load_aspect2id (
        aspect_id_file_path, word_id_mapping, w2v,300)


    # self.aspect_embedding = tf.Variable(aspect_embed, dtype=tf.float32, name='aspect_embedding')
