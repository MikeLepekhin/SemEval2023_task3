import argparse
import glob
import logging
import os.path
import pandas as pd
import shutil
import random
import torch

from data_processing import *
from models import *
from os import makedirs, remove
from sklearn.metrics import accuracy_score
from training import *
from tqdm import tqdm


def short_text(text):
    return ' '.join(str(text).split()[:500])

def text_batch_to_ids(text_list, tokenizer, indexer, vocab, device):
    MAX_TOKENS = 512
    
    text_fields = []
    padding_lengths = {}
    
    for text in text_list:
        text_fields.append(TextField(tokenizer.tokenize(short_text(text))[:MAX_TOKENS-2], {"bert_tokens": indexer}))
        text_fields[-1].index(vocab)
        
        for key, value in text_fields[-1].get_padding_lengths().items():
            padding_lengths[key] = max(value, padding_lengths[key] if key in padding_lengths else 0)
    
    result = text_fields[0].batch_tensors([text_field.as_tensor(padding_lengths)\
                                          for text_field in text_fields])
    
    result['bert_tokens']['token_ids'] = result['bert_tokens']['token_ids'].to(device)
    result['bert_tokens']['mask'] = result['bert_tokens']['mask'].to(device)
    result['bert_tokens']['type_ids'] = result['bert_tokens']['type_ids'].to(device)
    return result

def predict_probs_for_texts(text_list, batch_size, model, tokenizer, indexer, vocab, device, dropout=False):
    all_predict = []
    id_to_label = vocab.get_index_to_token_vocabulary('labels')
    
    if dropout:
        model.train()
        
    with torch.no_grad():
        for batch_start in range(0, len(text_list), batch_size):
            batch_end = min(len(text_list), batch_start + batch_size)
            padded_batch = text_batch_to_ids(text_list[batch_start:batch_end], tokenizer, indexer, vocab, device)
            batch_predict = model(padded_batch)['probs']
            all_predict.append(batch_predict.cpu().numpy())
                
    return np.concatenate(all_predict)

def predict_token_embeddings_for_texts(text_list, batch_size, model, tokenizer, indexer, vocab, device, dropout=False):
    all_predict = []
    id_to_label = vocab.get_index_to_token_vocabulary('labels')
    
    if dropout:
        model.train()
        
    all_predict = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(text_list), batch_size)):
            batch_end = min(len(text_list), batch_start + batch_size)
            padded_batch = text_batch_to_ids(text_list[batch_start:batch_end], tokenizer, indexer, vocab, device)
      
            batch_predict, mask = model.get_token_embeddings(padded_batch)
            #print(batch_predict)
            #print(mask)
            
            for text_id in range(len(batch_predict)):
                cur_predict = []
                
                for i in range(len(batch_predict[text_id])):
                    if mask[text_id][i]:
                        cur_predict.append(batch_predict[text_id][i].cpu().numpy())
                all_predict.append(cur_predict)
                
    return all_predict

def predict_embeddings_for_texts(text_list, batch_size, model, tokenizer, indexer, vocab, device, dropout=False):
    all_predict = []
    id_to_label = vocab.get_index_to_token_vocabulary('labels')
    
    if dropout:
        model.train()
        
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(text_list), batch_size)):
            batch_end = min(len(text_list), batch_start + batch_size)
            padded_batch = text_batch_to_ids(text_list[batch_start:batch_end], tokenizer, indexer, vocab, device)
            batch_predict = model.get_embedding(padded_batch)
            all_predict.append(batch_predict.cpu().numpy())
                
    return np.concatenate(all_predict)

def predict_labels_for_texts(text_list, batch_size, model, tokenizer, indexer, vocab, device, dropout=False):
    all_predict = None
    id_to_label = vocab.get_index_to_token_vocabulary('labels')
    
    if dropout:
        model.train()
    
    with torch.no_grad():
        for batch_start in range(0, len(text_list), batch_size):
            batch_end = min(len(text_list), batch_start + batch_size)
            padded_batch = text_batch_to_ids(text_list[batch_start:batch_end], tokenizer, indexer, vocab, device)
            batch_predict = model(padded_batch)['probs']
            if all_predict is None:
                all_predict = list(np.argmax(batch_predict.cpu().numpy(), axis=1))
            else:
                all_predict += list(np.argmax(batch_predict.cpu().numpy(), axis=1))
                
    all_predict = [id_to_label[target_id] for target_id in all_predict]
    return all_predict   

def predict_embeddings_file(filename, batch_size, model, tokenizer, indexer, vocab, device, dropout=False):
    return predict_embeddings_for_texts(
        pd.read_csv(filename).text.values, batch_size,
        model, tokenizer, indexer, vocab, device, dropout
    )

def predict_probs_file(filename, batch_size, model, tokenizer, indexer, vocab, device, dropout=False):
    return predict_probs_for_texts(
        pd.read_csv(filename).text.values, batch_size,
        model, tokenizer, indexer, vocab, device, dropout
    )

def predict_file(filename, batch_size, model, tokenizer, indexer, vocab, device):
    return predict_labels_for_texts(
        pd.read_csv(filename).text.values, batch_size,
        model, tokenizer, indexer, vocab, device
    )

def get_transformer_model_predictions(transformer_model, test_data_filename, model_dir,
                                      batch_size, cuda_device=-1, use_bert_pooler=False,
                                      verbose=True):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) transformer_model - model type (example: bert-base-cased),
    2) test_data_filename - name of the test data file in csv format,
    3) model_dir - directory where to save the model after training,
    4) batch_size - batch size,
    5) cuda_device - cuda device id on which the model trains 
    (if set to -1, the training performs on CPU),
    6) use_bert_pooler - indicates whether should we take the embedding
    of the first token as the text embedding,
    7) verbose - indicates whether the logger shows INFO and WARNING messages.
    '''
    
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
    
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    best_model = os.path.join(checkpoints_dir, 'best.th')
    vocab = Vocabulary().from_files(os.path.join(model_dir, 'vocab'))
    
    device = 'cpu' if cuda_device == -1 else 'cuda:' + str(cuda_device) 
    
    # initialize the model
    if use_bert_pooler:
        model = build_transformer_model(vocab, transformer_model).to(device)
    else:
        model = build_pool_transformer_model(vocab, transformer_model).to(device)
        
    # load the model weights
    model.load_state_dict(torch.load(best_model, map_location=device))
        
    tokenizer = PretrainedTransformerTokenizer(transformer_model)
    indexer = PretrainedTransformerIndexer(transformer_model)
    
    return predict_file(test_data_filename, batch_size, model, tokenizer, indexer, vocab, device)


def get_transformer_model_prob_predictions(transformer_model, test_data_filename, model_dir,
                                      batch_size, cuda_device=-1, use_bert_pooler=False,
                                      verbose=True, probs_filename=None, dropout=False):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) transformer_model - model type (example: bert-base-cased),
    2) test_data_filename - name of the test data file in csv format,
    3) model_dir - directory where to save the model after training,
    4) batch_size - batch size,
    5) cuda_device - cuda device id on which the model trains 
    (if set to -1, the training performs on CPU),
    6) use_bert_pooler - indicates whether should we take the embedding
    of the first token as the text embedding,
    7) verbose - indicates whether the logger shows INFO and WARNING messages,
    8) probs_filename - the name of the file where to save the predicted probabilities.
    '''
    
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
    
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    best_model = os.path.join(checkpoints_dir, 'best.th')
    vocab = Vocabulary().from_files(os.path.join(model_dir, 'vocab'))
    
    device = 'cpu' if cuda_device == -1 else 'cuda:' + str(cuda_device) 
    
    # initialize the model
    if use_bert_pooler:
        model = build_transformer_model(vocab, transformer_model).to(device)
    else:
        model = build_pool_transformer_model(vocab, transformer_model).to(device)
        
    # load the model weights
    model.load_state_dict(torch.load(best_model, map_location=device))
        
    tokenizer = PretrainedTransformerTokenizer(transformer_model)
    indexer = PretrainedTransformerIndexer(transformer_model)
    
    model_probs = predict_probs_file(
        test_data_filename, batch_size, model, tokenizer, indexer, vocab, device, dropout
    )
    
    vocab = Vocabulary().from_files(os.path.join(model_dir, 'vocab'))
    label_to_id = vocab.get_token_to_index_vocabulary('labels')
    #print(label_to_id)
    #all_labels = ['A1', 'A11', 'A12', 'A14', 'A16', 'A17', 'A4', 'A7', 'A8', 'A9']
    all_labels = ['Capacity_and_resources',
 'Crime_and_punishment',
 'Cultural_identity',
 'Economic',
 'External_regulation_and_reputation',
 'Fairness_and_equality',
 'Health_and_safety',
 'Legality_Constitutionality_and_jurisprudence',
 'Morality',
 'Policy_prescription_and_evaluation',
 'Political',
 'Public_opinion',
 'Quality_of_life',
 'Security_and_defense']
    
    '''all_labels = ['Appeal_to_Authority', 'Appeal_to_Fear-Prejudice',
       'Appeal_to_Hypocrisy', 'Appeal_to_Popularity', 'Appeal_to_Time',
       'Appeal_to_Values', 'Causal_Oversimplification',
       'Consequential_Oversimplification', 'Conversation_Killer', 'Doubt',
       'Exaggeration-Minimisation', 'False_Dilemma-No_Choice',
       'Flag_Waving', 'Guilt_by_Association', 'Loaded_Language',
       'Name_Calling-Labeling', 'Obfuscation-Vagueness-Confusion',
       'Questioning_the_Reputation', 'Red_Herring', 'Repetition',
       'Slogans', 'Straw_Man', 'Whataboutism']'''
    
    all_labels = ['reporting', 'opinion', 'satire']
    
    normalized_probs = np.zeros_like(model_probs).astype(float)

    for label_id, label in enumerate(all_labels):
        normalized_probs[:, label_id] = model_probs[:, label_to_id[label]]
    
    if probs_filename is not None:
        np.save(probs_filename, normalized_probs)
    
    return normalized_probs

def get_transformer_model_embeddings(transformer_model, test_data_filename, model_dir,
                                      batch_size, cuda_device=-1, use_bert_pooler=False,
                                      verbose=True, probs_filename=None, dropout=False):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) transformer_model - model type (example: bert-base-cased),
    2) test_data_filename - name of the test data file in csv format,
    3) model_dir - directory where to save the model after training,
    4) batch_size - batch size,
    5) cuda_device - cuda device id on which the model trains 
    (if set to -1, the training performs on CPU),
    6) use_bert_pooler - indicates whether should we take the embedding
    of the first token as the text embedding,
    7) verbose - indicates whether the logger shows INFO and WARNING messages,
    8) probs_filename - the name of the file where to save the predicted probabilities.
    '''
    
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
    
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    best_model = os.path.join(checkpoints_dir, 'best.th')
    vocab = Vocabulary().from_files(os.path.join(model_dir, 'vocab'))
    
    device = 'cpu' if cuda_device == -1 else 'cuda:' + str(cuda_device) 
    
    # initialize the model
    if use_bert_pooler:
        model = build_transformer_model(vocab, transformer_model).to(device)
    else:
        model = build_pool_transformer_model(vocab, transformer_model).to(device)
        
    # load the model weights
    model.load_state_dict(torch.load(best_model, map_location=device))
        
    tokenizer = PretrainedTransformerTokenizer(transformer_model)
    indexer = PretrainedTransformerIndexer(transformer_model)
    
    return predict_embeddings_file(
        test_data_filename, batch_size, model, tokenizer, indexer, vocab, device, dropout
    )

def evaluate_transformer_classifier(transformer_model, test_data_filename, model_dir, 
                                    batch_size, cuda_device=-1, use_bert_pooler=False):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) transformer_model - model type (example: bert-base-cased),
    2) test_data_filename - name of the test data file in csv format,
    3) model_dir - directory where to save the model after training,
    4) batch_size - batch size,
    5) cuda_device - cuda device id on which the model trains 
    (if set to -1, the training performs on CPU),
    6) use_bert_pooler - indicates whether should we take the embedding
    of the first token as the text embedding.
    
    '''
    
    model_predictions = get_transformer_model_predictions(
        transformer_model, test_data_filename, model_dir,
        batch_size, cuda_device, use_bert_pooler
    )
    calc_classifier_metrics(model_predictions, list(pd.read_csv(test_data_filename).target.values))

    
def get_transformer_classifier_accuracy(transformer_model, test_data_filename, model_dir, 
                                        batch_size, cuda_device=-1, use_bert_pooler=False,
                                        probs_filename=None):
    '''
    Trains a transformer-like model with usage of the AllenNLP framework.
    
    Parameters.
    1) transformer_model - model type (example: bert-base-cased),
    2) test_data_filename - name of the test data file in csv format,
    3) model_dir - directory where to save the model after training,
    4) batch_size - batch size,
    5) cuda_device - cuda device id on which the model trains 
    (if set to -1, the training performs on CPU),
    6) use_bert_pooler - indicates whether should we take the embedding
    of the first token as the text embedding,
    7) probs_filename - the file where the classifier probs will be solved.
    
    '''
    
    if probs_filename is not None:
        all_labels = ['A1', 'A11', 'A12', 'A14', 'A16', 'A17', 'A4', 'A7', 'A8', 'A9']
        
        model_predictions = np.argmax(
            get_transformer_model_prob_predictions(
                transformer_model, test_data_filename, model_dir, batch_size,
                cuda_device, use_bert_pooler, verbose=False, probs_filename=probs_filename
            ),
            axis=-1
        )
        model_predictions = [all_labels[label_id] for label_id in model_predictions]
    else:
        model_predictions = get_transformer_model_predictions(
            transformer_model, test_data_filename, model_dir,
            batch_size, cuda_device, use_bert_pooler, verbose=False,
        )
   
    return accuracy_score(
        list(pd.read_csv(test_data_filename).target.values),
        model_predictions
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a transformer model.')
    
    parser.add_argument('--transformer-model', type=str, help='model type (example: bert-base-cased)')
    parser.add_argument('--test-data-filename', type=str, help='name of the test data file in csv format')
    parser.add_argument('--model-dir', type=str, help='directory where to save the model after training')
    parser.add_argument(
        '--use-bert-pooler', type=bool, default=False, 
         help='indicates whether should we take the embedding of the first token as the text embedding'
    )
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cuda-device', type=int, default=-1)
   
    args = parser.parse_args()
    
    evaluate_transformer_classifier(
        args.transformer_model, args.test_data_filename, args.model_dir, 
        args.batch_size, args.cuda_device, args.use_bert_pooler
    )