import pickle
import GRU_pos_slice
import preparedata
import json


def main():
    #
    prog_para = {
        'n_hidden': 128,
        'n_input_char': 10000,
        'n_emb': 300,
        'n_pos': 50,
        'model_name': 'w2v_pos_slice',
        'pre_trained_w2v': 'google_emb.npy',

    }
    print prog_para

    [ind2word, word2ind] = pickle.load(open('vocal.pkl', 'rb'))
    [ind2pos, pos2ind] = pickle.load(open('pos_dict.pkl', 'rb'))
    model = GRU_pos_slice.GRU(n_hidden=prog_para['n_hidden'], n_input_char=prog_para['n_input_char'],
                            n_pos=prog_para['n_pos'], n_emb=prog_para['n_emb'])
    model.loadmodel(prog_para['model_name'])
    model.loadWordVector(prog_para['pre_trained_w2v'])

    label_char = preparedata.label_char

    #test_1, test_2, test_l, pos_1, pos_2 = preparedata.loadFile('dev_pdtb.json')
    #test_data = [[a1, a2, l, p1, p2] for a1, a2, l, p1, p2 in zip(test_1, test_2, test_l, pos_1, pos_2)]

    acc = 0.
    cnt = 0.

    fp = open('dev_pdtb.json','r')
    fw = open('dev_result.json','w')
    fp_word = fp.readlines()
    for fp_simple in fp_word:
        dict = json.loads(fp_simple)
        if dict['Type'] == 'Implicit' :
            cnt += 1
            arg1 = dict.get('Arg1').get('Word')
            arg2 = dict.get('Arg2').get('Word')
            sense = dict['Sense']
            pos1 = dict.get('Arg1').get('POS')
            pos2 = dict.get('Arg2').get('POS')

            t1 = [[preparedata.vectorize(w, word2ind), pos2ind[p]] for w, p in zip(arg1, pos1)]
            t2 = [[preparedata.vectorize(w, word2ind), pos2ind[p]] for w, p in zip(arg2, pos2)]

            pred = model.predict(t1, t2)

            if [label_char[pred]] == sense:
                print 'predict: %s, object: %s -------------' % (label_char[pred], sense)
                acc += 1
            else:
                print 'predict: %s, object: %s' % (label_char[pred], sense)

            dict['Sense'] = [label_char[pred]]
            print dict['Sense']
            fw.write(json.dumps(dict)+"\n")

    
    fw.close()
    print acc / cnt

    #for sample in test_data:
    #    arg1 = sample[0]
    #    arg2 = sample[1]
    #    sense = sample[2]
    #    pos1 = sample[3]
    #    pos2 = sample[4]
    #    t1 = [[preparedata.vectorize(w, word2ind), pos2ind[p]] for w, p in zip(arg1, pos1)]
    #    t2 = [[preparedata.vectorize(w, word2ind), pos2ind[p]] for w, p in zip(arg2, pos2)]

    #    pred = model.predict(t1, t2)
    #    if label_char[pred] == sense:
    #        print 'predict: %s, object: %s -------------' % (label_char[pred], sense)
    #        acc += 1
    #    else:
    #        print 'predict: %s, object: %s' % (label_char[pred], sense)



if __name__ == '__main__':
    main()
