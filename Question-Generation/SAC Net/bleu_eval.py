import nltk
# from rouge_metric.rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


def calc_score(candidate, refs):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    assert (len(candidate) == 1)
    assert (len(refs) > 0)
    prec = []
    rec = []
    beta = 1.2

    # split into tokens
    token_c = candidate[0].split(" ")

    for reference in refs:
        # split into tokens
        token_r = reference.split(" ")
        # compute the longest common subsequence
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs / float(len(token_c)))
        rec.append(lcs / float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)

    if (prec_max != 0 and rec_max != 0):
        score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
    else:
        score = 0.0
    return score


def get_bleu_rouge(json_data):
    bleus = []
    rouges = []
    for data in json_data:
        char_ref = [x for x in ''.join(data['real_ques'])]
        char_gene = [x for x in ''.join(data['generated'])]
        references = [char_ref]
        candidate = char_gene
        smooth = SmoothingFunction()
        bleu_score = nltk.translate.bleu_score.sentence_bleu(references, candidate,
                                                             weights=(0.25, 0.25, 0.25, 0.25),
                                                             smoothing_function=smooth.method1)

        rouge_score = calc_score([' '.join(data['generated'])], [' '.join(data['real_ques'])])
        bleus.append(bleu_score)
        rouges.append(rouge_score)

    return bleus, rouges


if __name__ == '__main__':
    data = {'real_ques': ['上海', '哪个', '医院', '骨科', '比较好'],
            'generated': ['上海', '哪家', '医院', '骨科', '好']}
    pred_dict = {'123456': ['结 果 只 有 个'],
                 '123': ['说 什 呢']}
    ref_dict = {'123456': ['结 果 有 一 个', '啊 是', '凄 凄 切 切 群'],
                '123': ['说 萨 达 萨 达 多', '说 什 么']}

    # metrics = calc_score(pred_dict, ref_dict)
    metrics = calc_score([' '.join(data['generated'])], [' '.join(data['real_ques'])])
    print(metrics)
