#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import copy
from collections import Counter
from tqdm import tqdm


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_paragraph_score(sample):
    """
    For each paragraph, compute the f1 score compared with the question
    Args:
        sample: a sample in the dataset.
    Returns:
        None
    Raises:
        None
    """
    question = sample["segmented_question"]
    for doc in sample['documents']:
        doc['segmented_paragraphs_scores'] = []
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score,
                                                              para_tokens,
                                                              [question])
            else:
                related_score = 0.0
            doc['segmented_paragraphs_scores'].append(related_score)


def dup_remove(doc):
    """
    For each document, remove the duplicated paragraphs
    Args:
        doc: a doc in the sample
    Returns:
        bool
    Raises:
        None
    """
    paragraphs_his = {}
    del_ids = []
    para_id = None
    if 'most_related_para' in doc:
        para_id = doc['most_related_para']
    doc['paragraphs_length'] = []
    for p_idx, (segmented_paragraph, paragraph_score) in \
            enumerate(zip(doc["segmented_paragraphs"], doc["segmented_paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        if para_id is not None and p_idx < para_id:
            prev_del_num += 1
        del doc["segmented_paragraphs"][p_idx - del_num]
        del doc["segmented_paragraphs_scores"][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1
    if len(del_ids) != 0:
        if 'most_related_para' in doc:
            doc['most_related_para'] = para_id - prev_del_num
        doc['paragraphs'] = []
        for segmented_para in doc["segmented_paragraphs"]:
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def paragraph_selection(sample, mode):
    """
    For each document, select paragraphs that includes as much information as possible
    Args:
        sample: a sample in the dataset.
        mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    Returns:
        None
    Raises:
        None
    """
    # predefined maximum length of paragraph
    MAX_P_LEN = 512
    # predefined splitter
    splitter = u'<splitter>'
    # topN of related paragraph to choose
    topN = 3
    doc_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            return
    for d_idx, doc in enumerate(sample['documents']):
        if 'segmented_paragraphs_scores' not in doc:
            continue
        status = dup_remove(doc)
        segmented_title = doc["segmented_title"]
        title_len = len(segmented_title)
        para_id = None
        if doc_id is not None:
            para_id = sample['documents'][doc_id]['most_related_para']
        # total_len = title_len + sum(doc['paragraphs_length'])
        total_len = sum(doc['paragraphs_length'])
        # add splitter
        para_num = len(doc["segmented_paragraphs"])
        total_len += para_num - 1
        if total_len <= MAX_P_LEN:
            incre_len = 0
            total_segmented_content = []
            for p_idx, segmented_para in enumerate(doc["segmented_paragraphs"]):
                if doc_id == d_idx and para_id > p_idx:
                    incre_len += len(segmented_para + [splitter])
                    total_segmented_content += segmented_para + [splitter]
                if doc_id == d_idx and para_id == p_idx:
                    total_segmented_content += segmented_para
            if doc_id == d_idx:
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0
            continue
        # find topN paragraph id
        para_infos = []
        for p_idx, (para_tokens, para_scores) in \
                enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens, para_scores, len(para_tokens), p_idx))
        para_infos.sort(key=lambda x: (-x[1], x[2]))
        topN_idx = []
        for para_info in para_infos[:topN]:
            topN_idx.append(para_info[-1])
        final_idx = []
        total_len = 0
        if doc_id == d_idx:
            if mode == "train":
                final_idx.append(para_id)
                total_len = doc['paragraphs_length'][para_id] + 1
        for id in topN_idx:
            if total_len > MAX_P_LEN:
                break
            if doc_id == d_idx and id == para_id and mode == "train":
                continue
            total_len += doc['paragraphs_length'][id] + 1
            final_idx.append(id)
        total_segmented_content = []
        final_idx.sort()
        incre_len = 0
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += doc['paragraphs_length'][id] + 1
                total_segmented_content += doc['segmented_paragraphs'][id] + [splitter]
            if doc_id == d_idx and id == para_id:
                total_segmented_content += doc['segmented_paragraphs'][id]

        if doc_id == d_idx:
            answer_start = incre_len + sample['answer_spans'][0][0]
            answer_end = incre_len + sample['answer_spans'][0][1]
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end
        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len - 1]
        doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0


if __name__ == "__main__":
    # change search/zhidao in file_name and output_file
    for model in ['train', 'dev', 'test']:
        file_name = 'dataset/' + model + '_preprocessed/zhidao.' + model + '.json'
        output_file = 'dataset/' + model + '_preprocessed/zhidao.' + model + '.selected.json'
        if "train" in file_name:
            mode = 'train'
        else:
            mode = 'test'
        count = 0
        with open(file_name, 'r') as f:
            with open(output_file, 'w') as w:
                for line in tqdm(f):
                    count += 1
                    if count % 10000 == 0:
                        print('selected', count, '...')
                    sample = json.loads(line, encoding='utf8')
                    compute_paragraph_score(sample)
                    paragraph_selection(sample, mode)
                    w.write(json.dumps(sample, ensure_ascii=False) + '\n')
