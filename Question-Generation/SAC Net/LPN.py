import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from layers.lstm_attention import LSTMAttentionDot, SoftDotAttention
from torch.nn import Embedding


class Encoder(nn.Module):
    def __init__(self, config, use_features=True):
        super(Encoder, self).__init__()
        self.config = config
        if use_features:
            input_size = config.embedding_size + 1
        else:
            input_size = config.embedding_size
        hidden_size = config.hidden_size // 2
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=config.lstm_layers, dropout=config.dropout,
                           bidirectional=True, batch_first=True)
        self.rnn.flatten_parameters()
    def forward(self, inputs):
        outputs, _ = self.rnn(inputs)
        return outputs


class TextFieldPredictor(nn.Module):
    def __init__(self, config, embeddings):
        super(TextFieldPredictor, self).__init__()
        self.config = config
        self.embeddings = embeddings
        self.encoder = Encoder(config)
        self.attention = SoftDotAttention(config.hidden_size)

    def forward_prepro(self, input, input_masks, answer_features=None):
        self.input = input  # [bs, len_c]
        self.input_masks = input_masks  # [bs, len_c]
        input_embeddings = self.embeddings(input)

        if answer_features is not None:
            unsqueezed_answer_features = torch.unsqueeze(answer_features, 2)  # [bs, len_c]->[bs, len_c, 1]
            self.input_embeddings = torch.cat((input_embeddings, unsqueezed_answer_features), 2)  # [bs, len_c, embedding_size+1]
        else:
            self.input_embeddings = input_embeddings

        self.lstm_embeddings = self.encoder(self.input_embeddings)  # [bs, len_c, embedding_size+1]->[bs, len_c, dim]

        return self.lstm_embeddings

    def forward_similarity(self, hidden_state):
        h_tilde, attentions = self.attention(hidden_state,
                                             self.lstm_embeddings, self.input_masks)
        return h_tilde, torch.log(attentions + 1e-6), self.input


class SoftmaxPredictor(nn.Module):
    def __init__(self, config):
        super(SoftmaxPredictor, self).__init__()
        self.config = config
        self.projection = nn.Linear(config.hidden_size, config.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden_state):
        return self.log_softmax(self.projection(hidden_state))


class LPN(nn.Module):
    def __init__(self, config):
        super(LPN, self).__init__()
        self.config = config
        self.init_predictors()
        self.init_base_lstm()

    def get_type(self):
        return self.model_type

    def init_predictors(self):
        self.embedder = self.get_embedder()
        self.combiner = self.get_combiner()
        self.text_field_predictor = TextFieldPredictor(self.config, self.embedder)
        self.softmax_predictor = SoftmaxPredictor(self.config)

    def get_embedder(self):
        if self.config.embedding_path is not None:
            embeddings = np.load(self.config.embedding_path).astype(np.float32)
            self.config.vocab_size = embeddings.shape[0]
            self.config.embedding_size = embeddings.shape[1]
            embedder = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),
                                                    freeze=False if self.config.trainable_embedding else True)
        else:
            embedder = nn.Embedding(self.config.vocab_size, self.config.embedding_size)
        return embedder

    def get_combiner(self):
        combiner = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.config.hidden_size, 2),
            nn.LogSoftmax(dim=-1))
        return combiner

    def init_base_lstm(self):
        self.base_lstm = LSTMAttentionDot(input_size=self.config.embedding_size,
                                          hidden_size=self.config.hidden_size,
                                          batch_first=True)

    def combine_predictions_single(self, context_tokens,
                                   predictor_probs,
                                   attentions,
                                   language_probs):

        max_attention_length = attentions.size(1)
        pad_size = self.config.vocab_size - max_attention_length
        batch_size = attentions.size(0)

        context_tokens_padding = Variable(torch.LongTensor(batch_size, pad_size).zero_()).cuda()
        attentions_padding = Variable(torch.zeros(batch_size, pad_size)).cuda() + -1e10
        stacked_context_tokens = torch.cat((context_tokens, context_tokens_padding), 1)

        softmax_probs = torch.exp(predictor_probs[:, 0])
        text_field_probs = torch.exp(predictor_probs[:, 1])

        stacked_attentions = torch.cat((attentions, attentions_padding), 1)
        attention_results = Variable(torch.zeros(batch_size, self.config.vocab_size)).cuda() + -1e10
        attention_results.scatter_(1, stacked_context_tokens, stacked_attentions)

        use_softmax_predictor = softmax_probs > text_field_probs
        use_softmax_predictor = use_softmax_predictor.unsqueeze(-1).float()
        combined_predictions = language_probs * use_softmax_predictor + attention_results * (1 - use_softmax_predictor)

        return combined_predictions

    def predict(self, input_token,
                context_tokens,
                context_mask,
                answer_features,
                end_token=3,
                max_length=40,
                min_length=3):
        """
        input_token: Input token to start with
        context_tokens: Context tokens to use
        Do greedy decoding using input token and context tokens
        """

        context_embeddings = self.text_field_predictor.forward_prepro(context_tokens, input_masks=context_mask,
                                                                      answer_features=answer_features)  # [bs, len_c, dim]

        state_shape = (1, self.config.hidden_size)
        h0 = c0 = Variable(context_embeddings.data.new(*state_shape).zero_())
        cur_states = (h0, c0)

        def step(input_token, states):
            input_token = input_token.unsqueeze(1)
            cur_input_embedding = self.embedder(input_token)  # [bs, dim]
            # [bs, dim], (h,c)
            hidden_states, new_states = self.base_lstm.forward(cur_input_embedding, states, context_embeddings,
                                                               context_mask)
            hidden_states = hidden_states.contiguous()
            reshaped_hidden_states = hidden_states.view(-1, hidden_states.size(-1))  # [bs, dim]
            predictor_probs = self.combiner(reshaped_hidden_states)  # [bs, 2]

            language_probs = self.softmax_predictor(reshaped_hidden_states)  # [bs, vocab_size]
            reshaped_language_probs = language_probs.view(-1, language_probs.size(-1))  # [bs, vocab_size]

            _, attentions, _ = self.text_field_predictor.forward_similarity(hidden_states)  # [bs, len_c]

            # [bs, vocab_size]
            combined_predictions = self.combine_predictions_single(context_tokens=context_tokens,
                                                                   predictor_probs=predictor_probs,
                                                                   attentions=attentions,
                                                                   language_probs=reshaped_language_probs)

            loss, token = torch.max(combined_predictions, 1)
            return loss, token, new_states

        loss, new_token, new_states = step(input_token, cur_states)

        batch_size = new_token.size(0)
        finish_num = torch.sum(new_token.data == end_token)
        predicted_tokens = []
        while len(predicted_tokens) < max_length:
            predicted_tokens.append(new_token)
            loss, new_token, new_states = step(new_token, new_states)
            finish_num += torch.sum(new_token.data == end_token)
            if finish_num == batch_size:
                predicted_tokens.append(new_token)
                if len(predicted_tokens) >= min_length:
                    break
        predicted_tokens = torch.stack(predicted_tokens, 1)  # [bs, len]

        return predicted_tokens

    def combine_predictions(self, context_tokens,  # [bs, len_c]
                            predictor_probs,  # [bs, len_q, 2]
                            attentions,  # [bs, len_q, len_c]
                            language_probs):  # [bs, len_q, vocab_size]

        # to batch second
        predictor_probs = predictor_probs.transpose(0, 1)
        attentions = attentions.transpose(0, 1)
        language_probs = language_probs.transpose(0, 1)

        max_attention_length = attentions.size(2)  # len_c
        pad_size = self.config.vocab_size - max_attention_length
        batch_size = attentions.size(1)
        seq_size = attentions.size(0)

        context_tokens_padding = Variable(torch.LongTensor(batch_size, pad_size).zero_(),
                                          requires_grad=False).cuda()
        attentions_padding = Variable(torch.zeros(batch_size, pad_size) + -1e10, requires_grad=False).cuda()
        stacked_context_tokens = torch.cat((context_tokens, context_tokens_padding), 1)  # [bs, vocab_size]

        total_attention_results = []
        softmax_probs = predictor_probs[:, :, 0]  # [len_q, bs]
        text_field_probs = predictor_probs[:, :, 1]  # [len_q, bs]

        replicated_softmax_probs = softmax_probs.unsqueeze(2)  # [len_q, bs, 1]
        replicated_text_field_probs = text_field_probs.unsqueeze(2)  # [len_q, bs, 1]

        dims = replicated_softmax_probs.size()

        expanded_softmax_probs = replicated_softmax_probs.expand(dims[0], dims[1],
                                                                 self.config.vocab_size)  # [len_q, bs, vocab_size]
        expanded_text_field_probs = replicated_text_field_probs.expand(dims[0], dims[1],
                                                                       max_attention_length)  # [len_q, bs, len_c]

        for i in range(0, seq_size):
            selected_text_field_probs = expanded_text_field_probs[i, :, :]
            selected_attention = attentions[i, :, :] + selected_text_field_probs
            # [len_q, bs, len_c+padding=vocab_size]
            stacked_attentions = torch.cat((selected_attention, attentions_padding), 1)

            attention_results = Variable(
                torch.zeros(batch_size, self.config.vocab_size) + -1e10).cuda()  # [bs, vocab_size]
            attention_results.scatter_(1, stacked_context_tokens, stacked_attentions)  # [bs, vocab_size]
            attention_results = torch.log(torch.exp(attention_results) /
                                          torch.sum(torch.exp(attention_results), -1, keepdim=True) *
                                          torch.exp(replicated_text_field_probs[i, :, :]) + 1e-6)
            total_attention_results.append(attention_results)

        concated_attention_results = torch.stack(total_attention_results, 0)  # [len_q, bs, vocab_size]
        final_probs = torch.log(torch.exp(concated_attention_results) +
                                torch.exp(language_probs + expanded_softmax_probs) + 1e-6)

        # return to batch first
        final_probs = final_probs.transpose(0, 1)  # [bs, len_q, vocab_size]

        return final_probs

    def forward(self, input_tokens, context_tokens, context_masks, answer_features, is_training=True):
        self.context_tokens = context_tokens
        self.context_embeddings = self.text_field_predictor.forward_prepro(context_tokens, context_masks,
                                                                           answer_features)
        self.input_embeddings = self.embedder(input_tokens)

        batch_size = input_tokens.size(0)
        token_length = input_tokens.size(1)

        state_shape = (batch_size, self.config.hidden_size)
        h0 = c0 = Variable(self.input_embeddings.data.new(*state_shape).zero_(), requires_grad=False)

        hidden_states, _ = self.base_lstm.forward(self.input_embeddings, (h0, c0),
                                                  self.context_embeddings, context_masks)  # [bs, len_q, dim]

        hidden_states = hidden_states.contiguous()
        reshaped_hidden_states = hidden_states.view(batch_size * token_length, -1)  # [bs*len_q, dim]
        predictor_probs = self.combiner(reshaped_hidden_states)  # [bs*len_q, 2]
        reshaped_predictor_probs = predictor_probs.view(batch_size, token_length,
                                                        predictor_probs.size(-1))  # [bs, len_q, 2]

        language_probs = self.softmax_predictor(reshaped_hidden_states)  # [bs*len_q, vocab_size]
        reshaped_language_probs = language_probs.view(batch_size, token_length,
                                                      language_probs.size(-1))  # [bs, len_q, vocab_size]

        attentions_list = []
        for i in range(0, token_length):
            _, attentions, _ = self.text_field_predictor.forward_similarity(hidden_states[:, i, :])
            attentions_list.append(attentions)  # [[bs, len_c],[bs, len_c],...]
        attentions_sequence = torch.stack(attentions_list, 1)  # [bs, len_q, len_c]

        # [bs, len_q, vocab_size]
        combined_predictions = self.combine_predictions(context_tokens=self.context_tokens,  # [bs, len_c]
                                                        predictor_probs=reshaped_predictor_probs,  # [bs, len_q, 2]
                                                        attentions=attentions_sequence,  # [bs, len_q, len_c]
                                                        language_probs=reshaped_language_probs)  # [bs, len_q, vocab_size]

        if not is_training:
            return combined_predictions
        else:
            # ques_ids [bs, len_q]
            # NO <START> for y
            y_true = input_tokens[:, 1:].contiguous()
            y_true = y_true.view(-1)  # [bs*(len_q-1)]
            combined_predictions = combined_predictions[:, :-1, :].contiguous()  # [bs, len_q-1, vocab_size]
            combined_predictions = combined_predictions.view(-1, self.config.vocab_size)  # [bs*(len_q-1), vocab_size]
            seq2seq_loss = nn.NLLLoss(ignore_index=0)(combined_predictions, y_true)
            return seq2seq_loss
