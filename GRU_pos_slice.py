import theano
import theano.tensor as T

import numpy
import pickle
import timeit

import os


class GRU(object):
    def __init__(self, n_hidden, n_input_char, n_emb, n_pos):

        # embedding
        self.iemb = theano.shared(name='iemb', borrow=True,
                                  value=0.1 * numpy.random.randn(n_input_char, n_emb).astype(
                                      theano.config.floatX))
        self.posv = theano.shared(name='posv', borrow=True,
                                  value=numpy.diag(numpy.ones(n_pos).astype(
                                      theano.config.floatX)))

        # encoder1
        self.w = theano.shared(name='w', borrow=True,
                               value=0.1 * numpy.random.randn(3, n_emb + n_hidden + n_pos, n_hidden).astype(
                                   theano.config.floatX))
        self.b = theano.shared(name='b', borrow=True,
                               value=0.1 * numpy.random.randn(3, n_hidden).astype(
                                   theano.config.floatX))

        # encoder2
        self.w2 = theano.shared(name='w2', borrow=True,
                                value=0.1 * numpy.random.randn(3, n_emb + n_hidden + n_pos, n_hidden).astype(
                                    theano.config.floatX))
        self.b2 = theano.shared(name='br2', borrow=True,
                                value=0.1 * numpy.random.randn(3, n_hidden).astype(
                                    theano.config.floatX))

        # mlp
        self.mlp_w = theano.shared(name='mlp_w', borrow=True,
                                   value=0.1 * numpy.random.randn(n_hidden * 2, n_hidden).astype(
                                       theano.config.floatX))
        self.mlp_b = theano.shared(name='mlp_b', borrow=True,
                                   value=0.1 * numpy.random.randn(n_hidden).astype(
                                       theano.config.floatX))

        self.output_w = theano.shared(name='output_w', borrow=True,
                                      value=0.1 * numpy.random.randn(n_hidden, 12).astype(
                                          theano.config.floatX))

        self.output_b = theano.shared(name='output_b', borrow=True,
                                      value=0.1 * numpy.random.randn(12).astype(
                                          theano.config.floatX))

        self.params = [self.w, self.b,
                       self.w2, self.b2,
                       self.mlp_w, self.mlp_b, self.output_w, self.output_b]

        l2_regularity = 0.00003
        self.l2mask = [numpy_floatX(l2_regularity), numpy_floatX(0),
                       numpy_floatX(l2_regularity), numpy_floatX(0),
                       numpy_floatX(l2_regularity), numpy_floatX(0)
                       ]

        def slice(W):
            return W[:n_emb], W[n_emb:n_emb + n_pos], W[n_emb + n_pos:]

        lr = T.scalar('lr', dtype=theano.config.floatX)
        arg1 = T.matrix('arg1', dtype='int64')
        arg2 = T.matrix('arg2', dtype='int64')
        label = T.scalar('label', dtype='int64')

        def encoder(word, h_tm1, w, b):
            x_t = self.iemb[word[0]]
            p_t = self.posv[word[1]]
            wr = slice(w[0])
            wz = slice(w[1])
            wh = slice(w[2])
            r_t = T.nnet.sigmoid(T.dot(x_t, wr[0]) + T.dot(p_t, wr[1]) + T.dot(h_tm1, wr[2]) + b[0])
            z_t = T.nnet.sigmoid(T.dot(x_t, wz[0]) + T.dot(p_t, wz[1]) + T.dot(h_tm1, wz[2]) + b[1])
            uh_tm1 = h_tm1 * r_t
            uh_t = T.tanh(T.dot(x_t, wh[0]) + T.dot(p_t, wh[1]) + T.dot(uh_tm1, wh[2]) + b[2])
            h_t = h_tm1 * z_t + uh_t * (1 - z_t)
            return h_t

        encode_process, _ = theano.scan(fn=encoder, sequences=arg1, outputs_info=dict(initial=T.zeros(n_hidden)),
                                        non_sequences=[self.w, self.b])
        arg1_code = encode_process[-1]

        encode_process, _ = theano.scan(fn=encoder, sequences=arg2, outputs_info=dict(initial=T.zeros(n_hidden)),
                                        non_sequences=[self.w2, self.b2], go_backwards=True)
        arg2_code = encode_process[-1]

        con = T.concatenate([arg1_code, arg2_code])
        internal = T.tanh(T.dot(con, self.mlp_w) + self.mlp_b)
        pred = T.nnet.softmax(T.dot(internal, self.output_w) + self.output_b)[0]

        nll = -T.log(pred[label])

        # SGD
        grad = T.grad(nll, self.params)
        grad_updates = [(param, param - lr * grad - l2_regular * param) for param, grad, l2_regular in
                        zip(self.params, grad, self.l2mask)]

        self.sentence_train = theano.function(inputs=[arg1, arg2, label, lr], outputs=nll,
                                              updates=grad_updates)
        self.sentence_error = theano.function(inputs=[arg1, arg2, label], outputs=nll)
        self.predict = theano.function(inputs=[arg1, arg2], outputs=T.argmax(pred))

    def savemodel(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for param in self.params:
            numpy.save(os.path.join(folder,
                                    param.name + '.npy'), param.get_value())

    def loadWordVector(self, file):
        if os.path.exists(file):
            self.iemb.set_value(numpy.load(file))
        else:
            print 'No Word vector Exist..'

    def loadmodel(self, folder):
        if os.path.isdir(folder):
            for param in self.params:
                param.set_value(numpy.load(os.path.join(folder,
                                                        param.name + '.npy')))
        else:
            print 'No Model Exist..'


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


def numpy_int64(data):
    return numpy.asarray(data, dtype='int64')


def main():
    prog_para = {
        'n_hidden': 128,
        'n_input_char': 10000,
        'n_emb': 300,
        'n_pos': 50,
        'model_name': 'w2v_pos_slice',
        'pre_trained_w2v': 'google_emb.npy',

        'print_freq': 1000,
        'num_epoch': 1000,
        'num_case_in_epoch': 100000
    }

    print prog_para

    print 'loading....'
    [train_data, test_data] = pickle.load(open('discourse.pkl', 'rb'))
    [train_pos, test_pos] = pickle.load(open('pos_data.pkl', 'rb'))
    train_data = [[
                      [[w, p] for w, p in zip(sentence[0], pos[0])],
                      [[w, p] for w, p in zip(sentence[1], pos[1])],
                      sentence[2]
                  ] for sentence, pos in zip(train_data, train_pos)]
    test_data = [[
                     [[w, p] for w, p in zip(sentence[0], pos[0])],
                     [[w, p] for w, p in zip(sentence[1], pos[1])],
                     sentence[2]
                 ] for sentence, pos in zip(test_data, test_pos)]

    print 'modeling....'
    model = GRU(n_hidden=prog_para['n_hidden'], n_input_char=prog_para['n_input_char'],
                n_pos=prog_para['n_pos'], n_emb=prog_para['n_emb'])
    model.loadmodel(prog_para['model_name'])
    model.loadWordVector(prog_para['pre_trained_w2v'])

    print 'trainning....'
    bestacc = 0.
    for epoch in range(prog_para['num_epoch']):
        epoch_tic = timeit.default_timer()
        numpy.random.shuffle(train_data)
        pos_split = len(train_data) / 10
        t_data, v_data = train_data[pos_split:], train_data[:pos_split]

        cnt = 0
        train_err = 0.
        for [arg1, arg2, y] in t_data:
            train_err += model.sentence_train(arg1, arg2, y, 0.01)
            cnt += 1
            if cnt % prog_para['print_freq'] == 0:
                print train_err / cnt
            if cnt == prog_para['num_case_in_epoch']:
                break

        # validate
        v_err = 0.
        for [arg1, arg2, y] in v_data:
            v_err += model.sentence_error(arg1, arg2, y)

        # test
        t_err = 0.
        acc = 0.
        for [arg1, arg2, y] in test_data:
            t_err += model.sentence_error(arg1, arg2, y)
            pred = model.predict(arg1, arg2)
            if pred == y:
                acc += 1

        if acc / len(test_data) > bestacc:
            model.savemodel(prog_para['model_name'])
            bestacc = acc / len(test_data)
        print '[trainning] epoch complete in %.2f, train_err=%.6f, validate_err=%.6f, test_err=%.6f, test_acc=%.6f,epoch=%d ' \
              % (timeit.default_timer() - epoch_tic, train_err / cnt, v_err / len(v_data), t_err / len(test_data),
                 acc / len(test_data), epoch)


if __name__ == '__main__':
    main()
