from collections import Counter
import collections
from tqdm import tqdm
import numpy as np
import json


# set train dev test examples in here
def search_words(examples):
    word_counter = Counter()
    for example in tqdm(examples):
        for token in example['question_tokens']:
            word_counter[token] += 1
        for para in example['passage_tokens']:
            for token in para:
                word_counter[token] += 1

    print('total tokens:', len(word_counter))

    return word_counter


def filter_words(word_counter, min_count=2):
    filtered_tokens = {token for token in word_counter if word_counter[token] >= min_count}

    print('after filtered with %d, remain %d tokens' % (min_count, len(filtered_tokens)))
    return filtered_tokens


class Tokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, word_counter=None,
                 vocab_load_file=None, embedding_load_file=None):
        # TODO: word_piece
        self.word_counter = word_counter
        self.embedding = None
        self.vocab = None
        self.ques_wp_over = 0
        self.cont_wp_over = 0
        if vocab_load_file is not None:
            with open(vocab_load_file, 'r') as f:
                self.vocab = json.load(f)
        if embedding_load_file is not None:
            self.embedding = np.load(embedding_load_file)
        if self.embedding is None or self.vocab is None:
            self.get_vocab_embedding(vocab_file)

    def get_vocab_embedding(self, vocab_file, size=8824330, vec_size=200):
        """Loads a vocabulary file into a dictionary."""
        self.vocab = collections.OrderedDict()
        self.vocab['--PAD--'] = 0
        self.vocab['<splitter>'] = 1
        self.vocab['--OOV--'] = 2
        self.vocab['<START>'] = 3
        self.vocab['<END>'] = 4

        self.embedding = [np.zeros(vec_size), np.zeros(vec_size),
                          (np.random.random(vec_size) - 0.5) / 5.0, (np.random.random(vec_size) - 0.5) / 5.0,
                          (np.random.random(vec_size) - 0.5) / 5.0]
        first_line = True
        with open(vocab_file, "r") as reader:
            for line in tqdm(reader, total=size):
                if first_line:
                    first_line = False
                    continue
                array = line.split()
                word = "".join(array[0:-vec_size])
                if word in self.vocab:
                    print(word, 'is in the vocab already!')
                    continue
                if word in self.word_counter:
                    vector = np.array(list(map(float, array[-vec_size:])))
                    self.vocab[word] = len(self.vocab)
                    self.embedding.append(vector)

        self.embedding = np.array(self.embedding)
        print('embedding shape:', self.embedding.shape)
        print('vocab size:', len(self.vocab))

        self.unk_set = self.word_counter - set(self.vocab.keys())
        print('unknown tokens:', len(self.unk_set))

    def convert_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['--OOV--'])
        return ids
