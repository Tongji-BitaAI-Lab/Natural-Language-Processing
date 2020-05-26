import argparse
import logging
import os
import pickle
from LPN_stroke import LPN
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from bleu_eval import get_bleu_rouge
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='CQG009_stroke', type=str, help="model name")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--save_checkpoints", default=5, type=int, help="epochs")
    parser.add_argument("--train_dir", default='dataset/preprocessed_data/train_features_all_part.pkl', type=str,
                        help="trainset dir")
    parser.add_argument("--dev_dir", default='dataset/preprocessed_data/dev_features_all.pkl', type=str,
                        help="devset dir")
    parser.add_argument("--test_dir", default='dataset/preprocessed_data/test_features_all.pkl', type=str,
                        help="testset dir")
    parser.add_argument("--vocab_dir", default='dataset/preprocessed_data/vocab.json', type=str, help="vocab dir")
    parser.add_argument("--embedding_path", default='dataset/preprocessed_data/embedding_mat.npy', type=str,
                        help="embedding dir")
    parser.add_argument('--stroke_path', type=str, default='dataset/preprocessed_data/stroke_mat.npy')
    parser.add_argument("--use_stroke", default=True, type=bool, help="use stroke")
    parser.add_argument("--trainable_embedding", default=True, type=bool, help="whether to train embedding")
    parser.add_argument("--hidden_size", default=256, type=int, help="hidden size")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
    parser.add_argument("--lstm_layers", default=2, type=int, help="lstm layers")
    parser.add_argument("--lr", default=5e-4, type=float, help="LR")
    parser.add_argument("--epochs", default=40, type=int, help="epochs")
    parser.add_argument("--show_num", default=50, type=int, help="show_num")
    parser.add_argument("--seed", default=666, type=int, help="set_seed")

    return parser.parse_args()


def make_fold(config):
    config.checkpoints_dir = os.path.join('models', config.name, 'model_weights')
    config.results_dir = os.path.join('models', config.name, 'results')
    os.makedirs(os.path.join('models', config.name), exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)


def get_dataloader(config, features, shuffle=False):
    question_ids = torch.tensor([f['ques_ids'][:32] for f in features], dtype=torch.long)
    context_ids = torch.tensor([f['context_ids'] for f in features], dtype=torch.long)
    context_masks = torch.tensor([f['context_mask'] for f in features], dtype=torch.uint8)
    answer_features = torch.tensor([f['answer_feat'] for f in features], dtype=torch.float)
    data = TensorDataset(question_ids, context_ids, context_masks, answer_features)
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=config.batch_size)

    return dataloader


def evaluate(test_dataloader, model, config, device, id2word_dict, epoch, save_flag=True):
    def convert_id2token(tokens):
        new_sentence = []
        for token in tokens:
            new_word = id2word_dict[token]
            if new_word == '<START>':
                continue
            elif new_word == '<END>':
                break
            else:
                new_sentence.append(new_word)
        return new_sentence

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    generated_questions = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        real_inputs, context_ids, context_masks, answer_features = batch
        question_ids = real_inputs[:, 0]
        predicted_tokens = model.predict(question_ids, context_ids, context_masks, answer_features)  # list[bs]
        for j, _ in enumerate(predicted_tokens):
            generated_questions.append({'real_ques': convert_id2token(real_inputs[j].detach().cpu().numpy()),
                                        'generated': convert_id2token(predicted_tokens[j].detach().cpu().numpy())})
    if save_flag == True:
        with open(os.path.join(config.results_dir, 'generated_questions_' + str(epoch + 1) + '.json'), 'w') as f:
            json.dump(generated_questions, f)
    bleus, rouges = get_bleu_rouge(generated_questions)
    mean_bleus = np.mean(bleus)
    mean_rouges = np.mean(rouges)
    model.train()

    return mean_bleus, mean_rouges


def save_config(config):
    config_file = open('models/' + config.name + '/config_file.txt', 'w')
    print('from model_answer import LPN', file=config_file)
    print('现在是seq2seq+answer_feature', file=config_file)
    print('config.name\t', config.name, file=config_file)
    print('config.use_stroke\t', config.use_stroke, file=config_file)
    print('config.hidden_size\t', config.hidden_size, file=config_file)
    print('config.dropout\t', config.dropout, file=config_file)
    print('config.lstm_layers\t', config.lstm_layers, file=config_file)
    print('config.lr\t', config.lr, file=config_file)
    print('config.batch_size\t', config.batch_size, file=config_file)
    print('config.epochs\t', config.epochs, file=config_file)
    print('config.save_checkpoints\t', config.save_checkpoints, file=config_file)
    print('config.show_num\t', config.show_num, file=config_file)
    print('config.seed\t', config.seed, file=config_file)
    print('trainable_embedding\t', config.trainable_embedding, file=config_file)
    config_file.close()


if __name__ == '__main__':
    config = get_config()
    make_fold(config)
    save_config(config)
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)

    logger.info('loading data...')
    with open(config.train_dir, 'rb') as f:
        train_features = pickle.load(f)
    with open(config.dev_dir, 'rb') as f:
        dev_features = pickle.load(f)
    with open(config.test_dir, 'rb') as f:
        test_features = pickle.load(f)
    with open(config.vocab_dir, 'r') as f:
        vocab = json.load(f)
    id2word_dict = {}
    for word in vocab:
        id2word_dict[vocab[word]] = word

    train_dataloader = get_dataloader(config, train_features, shuffle=True)
    dev_dataloader = get_dataloader(config, dev_features, shuffle=False)
    test_dataloader = get_dataloader(config, test_features, shuffle=False)

    logger.info('init model...')
    model = LPN(config)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    optimizer = optim.Adamax(model.parameters(), lr=config.lr)

    # training
    model.train()
    logger.info('start training...')
    bleus_rouges = []
    epoch_bleus = []
    loss_out = []
    info_out = []
    total_step = 0
    for epoch in range(config.epochs):
        total_loss = 0
        for step, batch in enumerate(list(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            # input_tokens 4, 64
            # context_tokens 4, 512
            # context_masks 4, 512
            # answer_features 4, 512
            question_ids, context_ids, context_masks, answer_features = batch
            loss = model(question_ids, context_ids, context_masks, answer_features, is_training=True)
            total_loss += torch.mean(loss).item()
            info_out.append([total_step, torch.mean(loss).item()])
            total_step = total_step + 1
            if (step + 1) % config.show_num == 0:
                logger.info('Epoch = %d/%d steps = %d/%d loss = %.5f' %
                            (epoch + 1, config.epochs,
                             step + 1, len(train_dataloader),
                             np.mean([loss_50[1] for loss_50 in info_out[-50:]])))
                loss_out.append([epoch + 1, step + 1, np.mean([loss_50[1] for loss_50 in info_out[-50:]])])
                df_loss_out = pd.DataFrame(loss_out, columns=['epoch', 'step', 'loss'])
                df_loss_out.to_csv('models/' + config.name + '/loss_show_num_'+str(config.show_num)+'_avg.txt', index=None)
                # (torch.mean(loss).item() / (step + 1))))
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            model.zero_grad()
            if (step + 1) % (1000 * 128 // config.batch_size) == 0:
                bleu, rouge = evaluate(dev_dataloader, model, config, device, id2word_dict, epoch, save_flag=False)
                bleus_rouges.append([step + 1, bleu, rouge, 'dev'])

                bleu, rouge = evaluate(test_dataloader, model, config, device, id2word_dict, epoch, save_flag=False)
                bleus_rouges.append([step + 1, bleu, rouge, 'test'])
                df_bleus_rouges = pd.DataFrame(bleus_rouges, columns=['step', 'bleu', 'bleus_rouges', 'mode'])
                df_bleus_rouges.to_csv(os.path.join('models/' + config.name + '/bleu_rouge_results.csv'), index=None)
        # validate
        logger.info('start validating...')
        bleu, rouge = evaluate(dev_dataloader, model, config, device, id2word_dict, epoch)
        bleus_rouges.append([total_step + 1, bleu, rouge, 'dev'])
        epoch_bleus.append(bleu)

        bleu, rouge = evaluate(test_dataloader, model, config, device, id2word_dict, epoch, save_flag=False)
        bleus_rouges.append([total_step + 1, bleu, rouge, 'test'])
        df_bleus_rouges = pd.DataFrame(bleus_rouges, columns=['step', 'bleu', 'bleus_rouges', 'mode'])
        df_bleus_rouges.to_csv(os.path.join('models/' + config.name + '/bleu_rouge_results.csv'), index=None)
        if (epoch + 1) % config.save_checkpoints == 0:
            torch.save(model.state_dict(), os.path.join(config.checkpoints_dir,
                                                        'checkpoint_' + str(epoch + 1) + '.pth'))
        if len(epoch_bleus) == 1:
            bleu_best = epoch_bleus[0]
        elif bleu_best <= epoch_bleus[-1]:
            torch.save(model.state_dict(), os.path.join(config.checkpoints_dir, 'checkpoint_best.pth'))
    np.savetxt(os.path.join('models', config.name, 'loss_out.txt'), info_out, fmt='%10s', delimiter=',')
