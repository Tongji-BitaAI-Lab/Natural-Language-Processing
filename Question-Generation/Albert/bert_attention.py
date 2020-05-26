#! -*- coding: utf-8 -*-

import json, os
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
import codecs, re
import pickle
from tqdm import tqdm
import random
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set cuda device

max_p_len = 256
max_q_len = 32
max_a_len = 32
max_qa_len = max_q_len + max_a_len
batch_size = 64
epochs = 40

# bert配置
config_path = '/home/liwei/data/albert_tiny_zh_google/albert_config_tiny_g.json'
checkpoint_path = '/home/liwei/data/albert_tiny_zh_google/albert_model.ckpt'
dict_path = '/home/liwei/data/albert_tiny_zh_google/vocab.txt'

# # 标注数据
# webqa_data = json.load(open('/root/qa_datasets/WebQA.json'))
# sogou_data = json.load(open('/root/qa_datasets/SogouQA.json'))



with open("/home/liwei/Text-Summarizer-Pytorch-master-1205/data/finished/bert_train_features.pkl", 'rb') as f:
    train_features = pickle.load(f)
with open("/home/liwei/Text-Summarizer-Pytorch-master-1205/data/finished/bert_dev_features.pkl", 'rb') as f:
    dev_features = pickle.load(f)
random.shuffle(dev_features)
with open("/home/liwei/Text-Summarizer-Pytorch-master-1205/data/finished/bert_test_features.pkl", 'rb') as f:
    test_features = pickle.load(f)

_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t, _ in sorted(_token_dict.items(), key=lambda s: s[1]):
    if t not in token_dict:
        if len(t) == 3 and (Tokenizer._is_cjk_character(t[-1])
                            or Tokenizer._is_punctuation(t[-1])):
            continue
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        for i in idxs:
            D = self.data[i]
            question = ''.join(D['question_tokens'])
            question = re.sub(u' |、|；|，', ',', question)[:max_q_len]
            answer = ''.join(D['passage_tokens'][D['start_position']: D['end_position']])
            answer = re.sub(u' |、|；|，', ',', answer)[:max_a_len]
            passage = ''.join(D['passage_tokens'])
            passage = re.sub(u' |、|；|，', ',', passage)
            qa_token_ids, qa_segment_ids = tokenizer.encode(
                answer, question, max_length=max_qa_len + 1)
            p_token_ids, p_segment_ids = tokenizer.encode(passage,
                                                          max_length=max_p_len)
            token_ids = p_token_ids + qa_token_ids[1:]
            segment_ids = p_segment_ids + qa_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


model = build_bert_model(
    config_path,
    checkpoint_path,
    application='seq2seq',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    model='albert'
)
model.summary()


y_in = model.input[0][:, 1:]  # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i: i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passages, topk=2, mode='extractive'):
    # 输入是一个question和对应的多个passage
    token_ids, segment_ids = [], []
    for passage in passages:
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids = tokenizer.encode(passage, max_length=max_p_len)[0]
        q_token_ids = tokenizer.encode(question, max_length=max_q_len + 1)[0]
        token_ids.append(p_token_ids + q_token_ids[1:])
        segment_ids.append([0] * len(token_ids[-1]))
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_q_len):  # 强制要求输出不超过max_q_len字
        _target_ids, _segment_ids = [], []
        # 篇章与候选答案组合
        for tids, sids in zip(token_ids, segment_ids):
            for t in target_ids:
                _target_ids.append(tids + t)
                _segment_ids.append(sids + [1] * len(t))
        _padded_target_ids = sequence_padding(_target_ids)
        _padded_segment_ids = sequence_padding(_segment_ids)
        _probas = model.predict([_padded_target_ids, _padded_segment_ids
                                 ])[..., 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _probas = [
            _probas[j, len(ids) - 1] for j, ids in enumerate(_target_ids)
        ]
        _probas = np.array(_probas).reshape((len(token_ids), topk, -1))
        if i == 0:
            _probas_argmax = _probas[:, 0].argmax(axis=1)
            _available_idxs = np.where(_probas_argmax != 0)[0]
            if len(_available_idxs) == 0:
                return ''
            else:
                _probas = _probas[_available_idxs]
                token_ids = [token_ids[j] for j in _available_idxs]
                segment_ids = [segment_ids[j] for j in _available_idxs]
        if mode == 'extractive':
            _zeros = np.zeros_like(_probas)
            _ngrams = {}
            for p_token_ids in token_ids:
                for k, v in get_ngram_set(p_token_ids, i + 1).items():
                    _ngrams[k] = _ngrams.get(k, set()) | v
            for j, t in enumerate(target_ids):
                _available_idxs = _ngrams.get(tuple(t), set())
                _available_idxs.add(token_dict['[SEP]'])
                _available_idxs = [k - 3 for k in _available_idxs]
                _zeros[:, j, _available_idxs] = _probas[:, j, _available_idxs]
            _probas = _zeros
        _probas = (_probas ** 2).sum(0) / (_probas.sum(0) + 1)
        _log_probas = np.log(_probas + 1e-6)
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def predict_to_file(data, filename, topk=2, mode='extractive'):
    """将预测结果输出到文件，方便评估
    """
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q_text = ''.join(d['question_tokens'])
            q_text = re.sub(u' |、|；|，', ',', q_text)[:max_q_len]
            a_text = ''.join(d['passage_tokens'][d['start_position']: d['end_position']])
            a_text = re.sub(u' |、|；|，', ',', a_text)[:max_a_len]
            p_texts = ''.join(d['passage_tokens'])
            p_texts = [re.sub(u' |、|；|，', ',', p_texts)]

            # q_text = d['question']
            # p_texts = [p['passage'] for p in d['passages']]
            a = gen_answer(a_text, p_texts, topk, mode)
            if a:
                s = u'%s\t%s\n' % (a, q_text)
            else:
                pass
            f.write(s)
            f.flush()


def evaluate(data):
    total, right = 0., 0.
    for [x_true, y_true] in data:
        y_pred = model.predict(x_true).argmax(axis=-1)
        print('x_true:', x_true)
        print('y_true:', y_true)
        print('y_pred:', y_pred)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


FILENAME = 'saved_result/bert_unilm/'
os.makedirs(FILENAME, exist_ok=True)
dev_generator = data_generator(dev_features, batch_size)
test_generator = data_generator(test_features, batch_size)


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10
        self.accs = []
        self.batch_loss = []

    def on_batch_end(self, epoch, logs=None):
        self.batch_loss.append(logs['loss'])


    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(FILENAME + 'best_model.weights')
        filename = FILENAME + 'epoch_' + str(epoch) + '.txt'
        predict_to_file(dev_features[:1], filename, topk=2, mode='generation')
        dev_acc = evaluate(dev_generator)
        test_accs = evaluate(test_generator)
        self.accs.append([epoch, dev_acc, test_accs, logs['loss']])
        df_accs = pd.DataFrame(self.accs, columns=['epoch', 'dev_acc', 'test_acc', 'loss'])
        df_accs.to_csv(FILENAME + 'acc.csv', index=None)

        df_loss = pd.DataFrame(self.batch_loss, columns=['loss'])
        df_loss.to_csv(FILENAME + 'loss.csv', index=None)
if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_features[:100], batch_size)

    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')
