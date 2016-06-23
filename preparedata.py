import json
import pickle
import numpy

class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict2object)

    def dict2object(self, d):
        # convert dict to object
        if '__class__' in d:
            class_name = d.pop('__class__')
            module_name = d.pop('__module__')
            module = __import__(module_name)
            class_ = getattr(module, class_name)
            args = dict((key.encode('ascii'), value) for key, value in d.items())  # get args
            inst = class_(**args)  # create new instance
        else:
            inst = d
        return inst

def BuildVocabulary(token_sentences, size):
    word_freq = {}
    for sentence in token_sentences:
        for word in sentence:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    vocab = sorted(word_freq.items(), key=lambda x: (x[1]), reverse=True)
    if size > len(vocab):
        size = len(vocab)
    print 'Saw %d distinct words, the most %dth freq word is %s, occur %d' % (
        len(vocab), size - 1, vocab[size - 2][0], vocab[size - 2][1])
    vocab = vocab[:size - 1]
    ind2word = ['<UNKNOW>'] + [pair[0] for pair in vocab]
    word2ind = dict([(w, i) for i, w in enumerate(ind2word)])
    return ind2word, word2ind

def replace(str):
    new = ''
    for i in str:
        if '0' <= i <= '9':
            new += '#'
        else:
            new += i
    return new

def loadFile(filename):
    sentences_arg1 = []
    sentences_arg2 = []
    sentences_pos1 = []
    sentences_pos2 = []
    label = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            data = MyDecoder().decode(line)
            sense = data.get('Sense')[0]
            if sense != 'EntRel':
                arg1 = [replace(t) for t in data.get('Arg1').get('Word')]
                arg2 = [replace(t) for t in data.get('Arg2').get('Word')]
                pos1 = data.get('Arg1').get('POS')
                pos2 = data.get('Arg2').get('POS')
                sentences_arg1.append(arg1)
                sentences_arg2.append(arg2)
                sentences_pos1.append(pos1)
                sentences_pos2.append(pos2)
                label.append(sense)
        file.close()
    return sentences_arg1, sentences_arg2, label, sentences_pos1, sentences_pos2


def vectorize(word, word2ind):
    if word in word2ind:
        return word2ind[word]
    else:
        return 0  # UNKNOW


def reformat(arg1, arg2, labels, word2ind, label_char):
    test = []
    for a1, a2, l in zip(arg1, arg2, labels):
        t1 = [vectorize(w, word2ind) for w in a1]
        t2 = [vectorize(w, word2ind) for w in a2]
        for ind, la in enumerate(label_char):
            if l == la:
                test.append([t1, t2, ind])
    return test


def BuildGoogleEmb(word2ind, gvector_filename):
    vectors = [numpy.zeros(300).astype(dtype=numpy.float32)] * len(word2ind)
    with open(gvector_filename) as file:
        lines = file.readlines()
        for line in lines[1:]:
            splits = line.split(' ')
            if splits[0] in word2ind:
                vectors[word2ind[splits[0]]] = [float(x) for x in splits[1:301]]
        file.close()
    for word in word2ind.keys():
        if len(vectors[word2ind[word]]) != 300:
            print word
    return numpy.asarray(vectors, dtype=numpy.float32)


label_char = ['Comparison.Concession', 'Comparison.Contrast', 'Contingency.Cause', 'Contingency.Pragmatic cause',
              'EntRel', 'Expansion.Alternative', 'Expansion.Conjunction', 'Expansion.Instantiation',
              'Expansion.List',
              'Expansion.Restatement', 'Temporal.Asynchronous', 'Temporal.Synchrony']


def tryput(dict, str):
    if str in dict:
        dict[str] += 1
    else:
        dict[str] = 1
    return dict


def buildgram2ind(gram, size):
    gram = sorted(gram.items(), key=lambda x: (x[1]), reverse=True)[:size]
    ind2gram = [pair[0] for pair in gram]
    gram2ind = dict([(w, i) for i, w in enumerate(ind2gram)])
    return gram2ind


def buildngramdict(data):
    ugram = {}
    bgram = {}
    tgram = {}

    for [s1, s2, _] in data:
        for w1 in s1:
            for w2 in s2:
                tryput(bgram, (w1, w2))
                tryput(tgram, (w1, w2, s1[0]))
                tryput(tgram, (w1, w2, s1[-1]))
    bgram = buildgram2ind(bgram, 1000)
    tgram = buildgram2ind(tgram, 1000)
    return ugram, bgram, tgram


def buildgramvec(gram, s1, s2):
    vector = [0] * 2000
    for w1 in s1:
        for w2 in s2:
            ind = getgramind(gram, (w1, w2), 0)
            if ind != -1:
                vector[ind] = 1
            ind = getgramind(gram, (w1, w2, s1[0]), 1)
            if ind != -1:
                vector[ind] = 1
            ind = getgramind(gram, (w1, w2, s1[-1]), 1)
            if ind != -1:
                vector[ind] = 1
    return vector


def getgramind(gram, str, n):
    if n == 0:
        off = 0
    else:
        off = 1000
    if str in gram[n]:
        return off + gram[n][str]
    else:
        return -1


def main():
    train_1, train_2, train_l, train_pos1, train_pos2 = loadFile('train_pdtb.json')
    test_1, test_2, test_l, test_pos1, test_pos2 = loadFile('dev_pdtb.json')

    # build vocabulary
    sentence = train_1 + train_2 + test_1 + test_2
    ind2word, word2ind, = BuildVocabulary(sentence, 10000)
    data = ind2word, word2ind
    pickle.dump(data, open('vocal.pkl', 'wb'))

    # # build pos dict
    # pos = train_pos1 + train_pos2 + test_pos1 + test_pos2
    # ind2pos, pos2ind = BuildVocabulary(pos, 50)
    # data = ind2pos, pos2ind
    # pickle.dump(data, open('pos_dict.pkl', 'wb'))
    # train_data = reformat(train_pos1, train_pos2, train_l, pos2ind, label_char)
    # test_data = reformat(test_pos1, test_pos2, test_l, pos2ind, label_char)
    # pickle.dump([train_data, test_data], open('pos_data.pkl', 'wb'))

    [ind2word, word2ind] = pickle.load(open('vocal.pkl', 'rb'))

    # # google word2vec
    # numpy.save(open('google_emb.npy', 'w'), BuildGoogleEmb(word2ind, 'GoogleWordVector/GVector.txt'))

    # build sentence vector
    train_data = reformat(train_1, train_2, train_l, word2ind, label_char)
    test_data = reformat(test_1, test_2, test_l, word2ind, label_char)
    pickle.dump([train_data, test_data], open('discourse.pkl', 'wb'))

if __name__ == '__main__':
    main()
