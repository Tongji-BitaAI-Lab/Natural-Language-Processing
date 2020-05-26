# -*- coding: utf-8 -*
import json
import pickle
from tqdm import tqdm
from collections import Counter
import numpy as np
from tokenization import Tokenizer, search_words, filter_words
import os


def precision_recall_f1(prediction, ground_truth):
    prediction_tokens = [t for t in prediction]
    ground_truth_tokens = [t for t in ground_truth]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return f1


def read_examples(input_file,
                  is_training,
                  max_para_num=5,
                  max_para_len=512,
                  max_ques_len=64,
                  max_answer_len=256,
                  max_test_doc_num=5):
    CNF = {'。', '，', '、', ',', '.', '!', '！', '?', '？', ' '}

    yesno_map_dict = {'not_yesno': -1,
                      'yes': 0,
                      'no': 1,
                      'depends': 2}

    examples = []
    over_para_num = 0
    over_para_len = 0
    over_ques_len = 0
    over_answer_len = 0
    no_answer = 0
    answer_wrong = 0
    no_opinion = 0
    lost_para = 0
    coverge_rate = 0
    answer_location_wrong = 0

    count = 0

    with open(input_file, "r", encoding='utf-8') as reader:

        for line in reader:
            entry = json.loads(line)
            count += 1
            if count % 10000 == 0:
                if 'search' in input_file:
                    print('search processed %d samples...' % count)
                else:
                    print('zhidao processed %d samples...' % count)

            if is_training:
                if len(entry['answer_spans']) == 0 or len(entry['fake_answers']) == 0 or 'answer_docs' not in entry:
                    no_answer += 1
                    continue
                if entry['answer_spans'][0][1] - entry['answer_spans'][0][0] + 1 >= max_answer_len:
                    over_answer_len += 1
                    continue
                if len(entry['segmented_question']) >= max_ques_len:
                    over_ques_len += 1
                    continue
                if len(entry['documents']) == 0 or len(entry['documents']) > max_para_num:
                    over_para_num += 1
                    continue

            question_id = entry["question_id"]
            question_type = entry["question_type"]
            fact_or_opinion = entry["fact_or_opinion"]
            question_tokens = entry["segmented_question"]
            passage_tokens = []
            start_position = None
            end_position = None
            yesno_answer = None
            ans_doc = None
            answer_tokens = None
            hit_in_dev = False
            if len(question_tokens) > max_ques_len:
                over_ques_len += 1

            if is_training:
                ans_doc = entry['answer_docs'][0]
                start_position = [ans_doc, entry['answer_spans'][0][0]]
                end_position = [ans_doc, entry['answer_spans'][0][1]]
                if start_position[1] > end_position[1]:
                    answer_location_wrong += 1
                    continue
            elif 'dev' in input_file:
                if len(entry['fake_answers']) == 0:
                    no_answer += 1
                    continue

            doc_infos = []
            validate_ = True

            for i_d, doc in enumerate(entry['documents']):
                if is_training:
                    most_related_para = doc['most_related_para']
                    add_para = doc['segmented_paragraphs'][most_related_para]
                    if i_d == ans_doc:
                        if end_position[1] >= max_para_len:
                            validate_ = False
                            over_para_len += 1
                            break
                        elif end_position[1] >= len(add_para):
                            answer_location_wrong += 1
                            end_position[1] = min(len(add_para) - 1, end_position[1])
                            start_position[1] = min(len(add_para) - 1, start_position[1])
                        answer_tokens = add_para[start_position[1]:end_position[1]]
                    passage_tokens.append(add_para)
                else:
                    para_infos = []
                    for i_p, para_tokens in enumerate(doc['segmented_paragraphs']):
                        if len(para_tokens) > max_para_len:
                            over_para_len += 1
                        common_with_question = Counter(question_tokens) & Counter(para_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(correct_preds) / len(entry['segmented_question'])
                        para_infos.append((para_tokens, recall_wrt_question, len(para_tokens), i_p))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    for para_info in para_infos[:1]:
                        doc_infos.append(para_info)
                    if 'dev' in input_file and not hit_in_dev:
                        if "".join(doc_infos[-1][0]).find(entry['fake_answers'][0]) != -1:
                            hit_in_dev = True
                        else:
                            for ans_segmented in entry['segmented_answers']:
                                ans_text = "".join(ans_segmented)
                                if "".join(doc_infos[-1][0]).find(ans_text) != -1:
                                    hit_in_dev = True
                                    break

            if not validate_:
                continue

            if hit_in_dev:
                coverge_rate += 1

            if not is_training:
                for doc_info in doc_infos[:max_test_doc_num]:
                    passage_tokens.append(doc_info[0])
            else:
                if ans_doc >= len(entry['documents']):
                    lost_para += 1
                    continue

                if len(answer_tokens) < 1:
                    answer_wrong += 1
                    continue

                validate_ = True
                while answer_tokens[0] in CNF:
                    start_position[1] += 1
                    answer_tokens = answer_tokens[1:]
                    if len(answer_tokens) < 1:
                        validate_ = False
                        break

                if not validate_:
                    answer_wrong += 1
                    continue

                origin_answer = "".join(answer_tokens)
                if entry['fake_answers'][0].find(origin_answer) == -1:
                    answer_wrong += 1
                    continue
                    # print(entry['fake_answers'][0], '!VS!', origin_answer)
                    # print('\n')

                if question_type == 'YES_NO':
                    if len(entry['yesno_answers']) == 1:
                        yesno_answer = entry['yesno_answers'][0]
                    else:  # 如果存在多个yesno，如果均相同则随便用，如果有不同根据answer选择
                        if len(entry['yesno_answers']) == 0:
                            no_opinion += 1
                            continue
                        elif all(ya == entry['yesno_answers'][0] for ya in entry['yesno_answers']):
                            yesno_answer = entry['yesno_answers'][0]
                        else:
                            match_score = 0
                            match_idx = 0
                            for a_idx, ans in enumerate(entry['answers']):
                                f1 = precision_recall_f1(ans, entry['fake_answers'])
                                if f1 > match_score:
                                    match_score = f1
                                    match_idx = a_idx
                            yesno_answer = entry['yesno_answers'][match_idx]
                else:
                    yesno_answer = 'not_yesno'

                yesno_answer = yesno_answer.lower()
                if yesno_answer == 'no_opinion':
                    no_opinion += 1
                    continue

                yesno_answer = yesno_map_dict[yesno_answer]

            examples.append({'question_id': question_id,
                             'question_type': question_type,
                             'fact_or_opinion': fact_or_opinion,
                             'question_tokens': question_tokens,
                             'passage_tokens': passage_tokens,
                             'yesno_answer': yesno_answer,
                             'answer_tokens': answer_tokens,
                             'start_position': start_position,
                             'end_position': end_position})

    print('over para num:', over_para_num)
    print('over para len:', over_para_len)
    print('over ques len:', over_ques_len)
    print('over answer len:', over_answer_len)
    print('no answer:', no_answer)
    print('answer wrong:', answer_wrong)
    print('no opinion:', no_opinion)
    print('lost para:', lost_para)
    print('answer location wrong:', answer_location_wrong)
    print('total remain:', len(examples))

    if not is_training and 'dev' in input_file:
        print('converge ratio:', coverge_rate / len(examples))

    return examples


def examples_to_features(examples, type, is_training,
                         tokenization,
                         max_para_num=5,
                         max_para_len=512,
                         max_ques_len=64):
    features = []
    for example in tqdm(examples):
        start_pos = None
        end_pos = None
        yesno_label = None

        p_ids_total = []
        p_mask_total = []
        for passage in example['passage_tokens']:
            p_ids = tokenization.convert_to_ids(passage[:max_para_len])
            p_mask = [1] * len(p_ids)
            if len(p_ids) < max_para_len:
                p_ids += ([tokenization.vocab['--PAD--']] * (max_para_len - len(p_ids)))
                p_mask += ([0] * (max_para_len - len(p_mask)))
            p_ids_total.append(p_ids)
            p_mask_total.append(p_mask)
        while len(p_ids_total) < max_para_num:
            p_ids_total.append([0] * max_para_len)
            p_mask_total.append([0] * max_para_len)

        p_ids_total = np.array(p_ids_total)
        p_ids = p_ids_total[example['start_position'][0]]
        p_mask_total = np.array(p_mask_total)
        p_mask = p_mask_total[example['start_position'][0]]

        q_ids = [tokenization.vocab['<START>']]
        q_ids.extend(tokenization.convert_to_ids(example['question_tokens'][:max_ques_len - 2]))
        q_ids.append(tokenization.vocab['<END>'])
        q_mask = [1] * len(q_ids)
        if len(q_ids) < max_ques_len:
            q_ids += ([tokenization.vocab['--PAD--']] * (max_ques_len - len(q_ids)))
            q_mask += ([0] * (max_ques_len - len(q_mask)))
        q_ids = np.array(q_ids)
        q_mask = np.array(q_mask)

        if is_training:
            start_pos = example['start_position'][1]
            end_pos = example['end_position'][1]
            yesno_label = example['yesno_answer']
        answer_feat = np.zeros(max_para_len)
        answer_feat[start_pos: end_pos + 1] = 1

        features.append(
            {'context_ids': p_ids,
             'context_mask': p_mask,
             'ques_ids': q_ids,
             'q_mask': q_mask,
             'answer_feat': answer_feat,
             'question_id': example['question_id'],
             'question_type': example['question_type'],
             'fact_or_opinion': example['fact_or_opinion'],
             'question_tokens': example['question_tokens'],
             'passage_tokens': example['passage_tokens'],
             'yesno_answer': example['yesno_answer'],
             'answer_tokens': example['answer_tokens'],
             'start_position': example['start_position'],
             'end_position': example['end_position']
             })
    if not os.path.exists('dataset/preprocessed_data/'):
        os.makedirs('dataset/preprocessed_data/')
    pickle.dump(features, open('dataset/preprocessed_data/' + type + '_features_all.pkl', 'wb'))


def get_examples(file_name, type, is_training):
    examples_search = read_examples(file_name + '/search.' + type + '.selected.json', is_training=is_training)
    examples_zhidao = read_examples(file_name + '/zhidao.' + type + '.selected.json', is_training=is_training)
    examples = examples_search + examples_zhidao
    if not os.path.exists('dataset/preprocessed_data/'):
        os.makedirs('dataset/preprocessed_data/')
    for i in range(len(examples)):
        for j, passage in enumerate(examples[i]['passage_tokens']):
            if '_' in passage:
                new_passage = passage[passage.index('_') + 1:]
                examples[i]['passage_tokens'][j] = new_passage
            elif '<splitter>' in passage:
                new_passage = passage[passage.index('<splitter>') + 1:]
                examples[i]['passage_tokens'][j] = new_passage
    json.dump(examples, open('dataset/preprocessed_data/' + type + '_examples.json', 'w'))

    print('######################################')
    print('example number:', len(examples))
    print('######################################')

    return examples


def example_wordpiece(examples, max_ques_len, max_para_len, max_char_len, total_vocab):
    def wordpiece(chars):
        if chars.lower() in total_vocab:
            return [chars.lower()]
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in total_vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return [chars]
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def wordpiece_question(tokens, max_q_len=64, max_char_len=50):
        output_tokens = []
        left_length = max_q_len - len(tokens)
        for token in tokens:
            if token in total_vocab or token == '<splitter>':
                output_tokens.append(token)
            else:
                if len(token) > max_char_len:
                    output_tokens.append(token)
                elif left_length > 0:
                    sub_tokens = wordpiece(token)
                    left_length -= (len(sub_tokens) - 1)
                    if left_length >= 0:
                        output_tokens.extend(sub_tokens)
                    else:
                        left_length += (len(sub_tokens) - 1)
                        output_tokens.append(token)
                else:
                    output_tokens.append(token)
        return output_tokens

    def wordpiece_context(tokens, ques_tokens_set, max_p_len=512, max_char_len=50, start_pos=None, end_pos=None):
        output_tokens = []
        for i_t, token in enumerate(tokens):
            if token == '<splitter>' or token in total_vocab or token in ques_tokens_set:  # 确保该单词未wordpiece没在ques里出现过
                output_tokens.append(token)
            else:
                if len(token) > max_char_len:
                    output_tokens.append(token)
                elif end_pos is None or end_pos[1] < max_p_len:
                    sub_tokens = wordpiece(token)
                    if start_pos is not None and end_pos is not None:
                        old_start_pos = start_pos[1]
                        old_end_pos = end_pos[1]
                        if i_t < start_pos[1]:  # token在答案起点前，起点和终点都+
                            start_pos[1] += (len(sub_tokens) - 1)
                            end_pos[1] += (len(sub_tokens) - 1)
                        elif start_pos[1] <= i_t <= end_pos[1]:  # token在答案中间，终点+，起点不变
                            end_pos[1] += (len(sub_tokens) - 1)
                        # token在答案后面，不影响答案

                        # 如果答案还在范围内则可以wordpiece
                        if end_pos[1] < max_p_len:
                            output_tokens.extend(sub_tokens)
                        else:
                            start_pos[1] = old_start_pos
                            end_pos[1] = old_end_pos
                            output_tokens.append(token)
                    else:
                        output_tokens.extend(sub_tokens)
                else:
                    output_tokens.append(token)

        if start_pos is not None and end_pos is not None:
            return output_tokens, start_pos, end_pos
        else:
            return output_tokens

    for example in tqdm(examples):
        example['question_tokens'] = wordpiece_question(example['question_tokens'],
                                                        max_q_len=max_ques_len, max_char_len=max_char_len)
        ques_token_set = set(example['question_tokens'])
        for i_p in range(len(example['passage_tokens'])):
            if example['start_position'] is not None and i_p == example['start_position'][0]:
                st = example['start_position']
                ed = example['end_position']
            else:
                st = None
                ed = None
            new_results = wordpiece_context(example['passage_tokens'][i_p],
                                            ques_token_set,
                                            max_p_len=max_para_len,
                                            max_char_len=max_char_len,
                                            start_pos=st,
                                            end_pos=ed)
            if st is None:
                example['passage_tokens'][i_p] = new_results
            else:
                example['passage_tokens'][i_p], example['start_position'], example['end_position'] = new_results

    return examples


if __name__ == '__main__':
    train_examples = get_examples('dataset/train_preprocessed', 'train', is_training=True)
    dev_examples = get_examples('dataset/dev_preprocessed', 'dev', is_training=True)
    # test_examples = get_examples('/users/caochenjie/DuReader/dataset/test1_preprocessed/test1set', 'test1', is_training=True)

    # with open('dataset/preprocessed_data/train_examples.json', 'r') as f:
    #     train_examples = json.load(f)
    # with open('dataset/preprocessed_data/dev_examples.json', 'r') as f:
    #     dev_examples = json.load(f)
    # with open('dataset/preprocessed_data/test1_examples.json', 'r') as f:
    #     test_examples = json.load(f)
    f = open('dataset/total_vocab.pkl', 'rb')
    total_vocab = f.read().decode()
    train_examples = example_wordpiece(train_examples, max_ques_len=64, max_para_len=512, max_char_len=32,
                                       total_vocab=total_vocab)
    dev_examples = example_wordpiece(dev_examples, max_ques_len=64, max_para_len=512, max_char_len=32,
                                     total_vocab=total_vocab)
    # test_examples = example_wordpiece(test_examples, max_ques_len=64, max_para_len=512, max_char_len=32,
    #                                   total_vocab=total_vocab)

    # word_counter = search_words(train_examples + dev_examples + test_examples)
    word_counter = search_words(train_examples + dev_examples)
    word_counter = filter_words(word_counter, min_count=18)
    tokenization = Tokenizer(vocab_file='/home/liwei/data/Tencent_AILab_ChineseEmbedding.txt',
                             word_counter=word_counter)
    if not os.path.exists('dataset/preprocessed_data/'):
        os.makedirs('dataset/preprocessed_data/')
    with open('dataset/preprocessed_data/vocab.json', 'w') as w:
        json.dump(tokenization.vocab, w, indent=4)
    np.save('dataset/preprocessed_data/embedding_mat.npy', tokenization.embedding)
    with open('/home/liwei/data/cw2vec.txt', 'r') as f:
        stroke_data = f.readlines()
    word2embedding = {}
    for i in stroke_data:
        word = i.split(' ')[0]
        embedding = list(map(float, i.split(' ')[1:]))
        word2embedding[word] = embedding
    stroke_embedding = []
    vec_size = 200
    for word in tokenization.vocab:
        UNK_embedding = (np.random.random(vec_size) - 0.5) / 5.0
        if word not in word2embedding:
            stroke_embedding.append(UNK_embedding)
        else:
            embedding_list = []
            for char in word:
                if char in word2embedding:
                    embedding_list.append(word2embedding[char])
                else:
                    embedding_list.append(UNK_embedding)
            stroke_embedding.append(np.mean(embedding_list, axis=0))
    np.save('dataset/preprocessed_data/stroke_mat.npy', stroke_embedding)

    examples_to_features(train_examples, type='train', is_training=True, max_para_len=512,
                         tokenization=tokenization)
    examples_to_features(dev_examples, type='dev', is_training=True, max_para_len=512,
                         tokenization=tokenization)
    # examples_to_features(test_examples, type='test1', is_training=True, max_para_len=512,
    #                      tokenization=tokenization)
