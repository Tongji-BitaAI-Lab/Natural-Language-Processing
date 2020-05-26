import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Vocab_token:
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


class Vocab_stroke:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = {
            ' ': 0, '{': 1, '}': 2,
            '点': 3, '横': 4, '横钩': 5, '横撇': 6, '横撇弯钩': 7, '横斜钩': 8,
            '横折': 9, '横折竖钩': 10, '横折提': 11, '横折弯': 12, '横折弯钩': 13,
            '横折折': 14, '横折折撇': 15, '横折折折': 16, '横折折折钩': 17, '捺': 18,
            '撇': 19, '撇点': 20, '撇折': 21, '竖': 22, '竖钩': 23, '竖提': 24, '竖弯': 25,
            '竖弯横钩': 26, '竖折': 27, '竖折撇': 28, '竖折折': 29, '竖折折钩': 30, '提': 31,
            '弯钩': 32, '卧钩': 33, '斜钩': 34, 'a': 35, 'b': 36, 'c': 37, 'd': 38, 'e': 39, 'f': 40, 'g': 41, 'h': 42,
            'i': 43, 'j': 44,
            'k': 45, 'l': 46, 'm': 47, 'n': 48, 'o': 49, 'p': 50, 'q': 51, 'r': 52, 's': 53, 't': 54, 'u': 55, 'v': 56,
            'w': 57, 'x': 58,
            'y': 59, 'z': 60, '1': 61, '2': 62, '3': 63, '4': 64, '5': 65, '6': 66, '7': 67, '8': 68, '9': 69, '0': 70,
            '!': 71, '@': 72, '#': 73, '$': 74, '%': 75, '^': 76, '&': 77, '*': 78, '(': 79, ')': 80, '_': 81, '+': 82,
            '.': 83
        }
        self._index2token = index2token or []
        self.word_stroke_dict = self.w2s()

    def is_Chinese(self, word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def w2s(self):
        word2stroke = {}
        with open('./data/preprocess_strokes.txt') as f:
            lines = f.readlines()
            for line in (lines):
                word = line[0]
                stroke = line[2:-1].split(',')
                word2stroke[word] = stroke
        return word2stroke

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def _flatten(myList):
    output = []
    for sublist in myList:
        for ele in sublist:
            output.append(ele)
    return output


def conv2d(input_, w, b, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.constant(w, shape=w.shape)
        b = tf.constant(b, shape=b.shape)

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def tdnn(input_, kernels, kernel_features, W, B, scope='TDNN'):
    '''
    :input:           input float tensor of shape [(一共有多少字) , 每个字拆成几笔画 , 每个笔画多少维度]
    :kernels:         in_wides
    :kernel_features: 输出多少维度
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]

    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size, w, b in zip(kernels, kernel_features, W, B):
            reduced_length = max_word_length - kernel_size + 1

            conv = conv2d(input_, w, b, name="kernel_%d" % kernel_size)
            # print('>>> Finish conv2d')
            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

    return output


def getVec(words):
    output = []
    temps = []
    for key in tqdm(words):
        if len(key) == 1:
            if key in char_vocab.word_stroke_dict:
                stroke = char_vocab.word_stroke_dict[key]
            else:
                stroke = char_vocab.word_stroke_dict['^']
            temp = ['{'] + stroke
            temp = temp[:72]
            temp = temp + ['}']

        else:
            strokes = []
            for char in key:
                if char in char_vocab.word_stroke_dict:
                    strokes.append(char_vocab.word_stroke_dict[char])
                else:
                    strokes.append(char_vocab.word_stroke_dict['^'])
                temp = _flatten(strokes)
                temp = ['{'] + temp
                temp = temp[:72]
                temp = temp + ['}']
        temps.append(temp)
        stroke_index = [char_vocab._token2index[i] if i in char_vocab._token2index else char_vocab._token2index['^'] for
                        i in temp]
        stroke_index_out = list(np.zeros([73], dtype=int))
        for i, j in enumerate(stroke_index):
            stroke_index_out[i] = j
        output.append(stroke_index_out)
    return temps, output, words


def highway(input_, size, bias=-2.0, f=tf.nn.relu, scope='Highway', LW=[], LB=[], GW=[], GB=[]):
    with tf.variable_scope('Highway0'):
        idx = 0
        g = f(linear(input_, size, LW, LB, idx=0, scope='highway_lin_%d' % idx))

        t = tf.sigmoid(gate(input_, size, GW, GB, idx=0, scope='highway_gate_%d' % idx))

        output = t * g + (1. - t) * input_
        input_ = output

    with tf.variable_scope('Highway1'):
        idx = 1
        g = f(linear(input_, size, LW, LB, idx=1, scope='highway_lin_%d' % idx))
        t = tf.sigmoid(gate(input_, size, GW, GB, idx=1, scope='highway_gate_%d' % idx) + bias)
        output = t * g + (1. - t) * input_

    return output


def linear(input_, output_size, LW, LB, idx, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.constant(LW[idx], shape=LW[idx].shape)
        bias_term = tf.constant(LB[idx], shape=LB[idx].shape)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def gate(input_, output_size, GW, GB, idx, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.constant(GW[idx], shape=GW[idx].shape)
        bias_term = tf.constant(GB[idx], shape=GB[idx].shape)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def batch_iter(x, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x[start_id:end_id]


if __name__ == '__main__':

    path = './dataset/preprocessed_data/vocab.json'

    with open(path, 'r') as f:
        word_dictionary = json.load(f)

    with open('/users/caochenjie/cn_stroke/character-aware-neural-language-models/char_vocab.data', 'rb') as f:
        char_vocab = pickle.load(f)
    with open('/users/caochenjie/cn_stroke/character-aware-neural-language-models/word_vocab.data', 'rb') as f:
        word_vocab = pickle.load(f)

    w1 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w1.npy')
    w2 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w2.npy')
    w3 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w3.npy')
    w4 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w4.npy')
    w5 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w5.npy')
    w6 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w6.npy')
    w7 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/w7.npy')
    b1 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b1.npy')
    b2 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b2.npy')
    b3 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b3.npy')
    b4 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b4.npy')
    b5 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b5.npy')
    b6 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b6.npy')
    b7 = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/b7.npy')
    stroke = np.load('/users/caochenjie/cn_stroke/character-aware-neural-language-models/stroke.npy')

    highway_lin0_matrix = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_lin0_matrix.npy')
    highway_lin1_matrix = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_lin1_matrix.npy')
    highway_lin0_bias = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_lin0_bias.npy')
    highway_lin1_bias = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_lin1_bias.npy')
    highway_gate0_matrix = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_gate0_matrix.npy')
    highway_gate1_matrix = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_gate1_matrix.npy')
    highway_gate0_bias = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_gate0_bias.npy')
    highway_gate1_bias = np.load(
        '/users/caochenjie/cn_stroke/character-aware-neural-language-models/para/highway_gate1_bias.npy')

    word_dictionary = {k: v for k, v in word_dictionary.items() if k not in {'--PAD--', '<splitter>', '--OOV--'}}
    strokes, wordRepressions, words = getVec(word_dictionary.keys())

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    output2 = []
    input_x = tf.placeholder(tf.int32, [None, 73], name='input_x')
    Stroke = tf.constant(stroke, dtype=tf.float32, shape=stroke.shape)
    W = [w1, w2, w3, w4, w5, w6, w7]
    B = [b1, b2, b3, b4, b5, b6, b7]
    LW = [highway_lin0_matrix, highway_lin1_matrix]
    LB = [highway_lin0_bias, highway_lin1_bias]
    GW = [highway_gate0_matrix, highway_gate1_matrix]
    GB = [highway_gate0_bias, highway_gate1_bias]

    wordRepression = batch_iter(wordRepressions, 32)
    embed = tf.nn.embedding_lookup(Stroke, input_x)
    output = tdnn(embed, [1, 2, 3, 4, 5, 6, 7], [50, 100, 150, 200, 200, 200, 200], W, B)
    input_ = output
    matrix_l0 = tf.constant(LW[0], shape=LW[0].shape)
    bias_term_l0 = tf.constant(LB[0], shape=LB[0].shape)
    g0 = tf.nn.relu(tf.matmul(input_, tf.transpose(matrix_l0)) + bias_term_l0)
    matrix_g0 = tf.constant(GW[0], shape=GW[0].shape)
    bias_term_g0 = tf.constant(GB[0], shape=GB[0].shape)
    t0 = tf.sigmoid(tf.matmul(input_, tf.transpose(matrix_g0)) + bias_term_g0)
    output_ = t0 * g0 + (1. - t0) * input_
    matrix_l1 = tf.constant(LW[1], shape=LW[1].shape)
    bias_term_l1 = tf.constant(LB[1], shape=LB[1].shape)
    g1 = tf.nn.relu(tf.matmul(output_, tf.transpose(matrix_l1)) + bias_term_l1)
    matrix_g1 = tf.constant(GW[1], shape=GW[1].shape)
    bias_term_g1 = tf.constant(GB[1], shape=GB[1].shape)
    t1 = tf.sigmoid(tf.matmul(output_, tf.transpose(matrix_g1)) + bias_term_g1)
    output_ = t1 * g1 + (1. - t1) * output_
    for batch_data in tqdm(wordRepression):
        temp = sess.run(output_, feed_dict={input_x: batch_data})
        output2.append(temp)

    result = [np.zeros(1100), np.zeros(1100), (np.random.random(1100) - 0.5) / 5.0]
    for vecs in tqdm(output2):
        for vec in vecs:
            result.append(vec.tolist())
    print('>>> word_dictionary.shape', len(word_dictionary))
    print('>>> result.shape', np.shape(result))
    np.save('dataset/preprocessed_data/stroke_result.npy', result)
