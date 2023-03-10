from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder, ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler, LstmSeq2VecEncoder, CnnEncoder, BagOfEmbeddingsEncoder, ClsPooler
from allennlp.nn import util
from allennlp.predictors import TextClassifierPredictor
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.training.metrics import FBetaMeasure

import numpy as np
import pandas as pd
import torch
from typing import Dict, Iterable, List, Tuple
from DeBERTa import deberta


class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        
    def get_embedding(self,
                text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        result = self.encoder(embedded_text, mask)
        #print(result.shape)
        return result
    
    def get_token_embeddings(self,
                text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        return embedded_text, mask
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label)
            self.accuracy(logits, label)
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

class Seq2SeqSegmenter(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 bos_id=0,
                 eos_id=2):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.gru = torch.nn.GRU(embedder.get_output_dim(), embedder.get_output_dim(), batch_first=True)
        self.classifier = torch.nn.Linear(embedder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.bos_id = bos_id
        self.eos_id = eos_id
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        
        
        batch_size = embedded_text.shape[0]
        num_tokens = embedded_text.shape[1]
        emb_size = embedded_text.shape[2]
        total_embed = []
        total_label = []
        #print(text, flush=True)
        #print(embedded_text, flush=True)
        #print(label, flush=True)
        
        if label is not None:
            for i in range(batch_size):
                label_ptr = 0
                cur_embed = []
                
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        cur_embed.append(torch.unsqueeze(embedded_text[i, j, :], 0))
                        
                        total_label.append(label[i][label_ptr])
                        label_ptr += 1
                cur_embed = torch.cat(cur_embed, dim=0)
                cur_embed = torch.unsqueeze(cur_embed, 0)
                
                output, h = self.gru(cur_embed)
                total_embed.append(output[0, :])
            total_embed = torch.cat(total_embed, dim=0)
        else:
            for i in range(batch_size):
                label_ptr = 0
                cur_embed = []
                
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        cur_embed.append(torch.unsqueeze(embedded_text[i, j, :], 0))
                        label_ptr += 1
                cur_embed = torch.cat(cur_embed, dim=0)
                cur_embed = torch.unsqueeze(cur_embed, 0)
                    
                output, h = self.gru(cur_embed)
                total_embed.append(output[0, :])
            total_embed = torch.cat(total_embed, dim=0)
                    
 
        logits = self.classifier(total_embed)
        
        if label is not None:
            total_label = torch.Tensor(total_label).long().to(logits.device)
        
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, total_label)
            self.accuracy(logits, total_label)
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    
class SimpleSegmenter(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 bos_id=0,
                 eos_id=2):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(embedder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.bos_id = bos_id
        self.eos_id = eos_id
        
    def get_embedding(self,
                text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        result = self.encoder(embedded_text, mask)
        #print(result.shape)
        return result
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        
        
        batch_size = embedded_text.shape[0]
        num_tokens = embedded_text.shape[1]
        emb_size = embedded_text.shape[2]
        total_embed = []
        total_label = []
        #print(text, flush=True)
        #print(embedded_text, flush=True)
        #print(label, flush=True)
        
        if label is not None:
            for i in range(batch_size):
                label_ptr = 0
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        total_embed.append(embedded_text[i, j, :])
                        total_label.append(label[i][label_ptr])
                        label_ptr += 1
        else:
            for i in range(batch_size):
                label_ptr = 0
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        total_embed.append(embedded_text[i, j, :])
                        label_ptr += 1
                    
        total_embed = torch.cat(total_embed).reshape((-1, emb_size))
        logits = self.classifier(total_embed)
        
        if label is not None:
            total_label = torch.Tensor(total_label).long().to(logits.device)
        
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, total_label)
            self.accuracy(logits, total_label)
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

class BinarySegmenter(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 bos_id=0,
                 eos_id=2,
                 coeff=0.5):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(embedder.get_output_dim(), num_labels)        
        self.binary_classifier = torch.nn.Linear(embedder.get_output_dim(), 2)

        self.accuracy = CategoricalAccuracy()
        self.binary_accuracy = CategoricalAccuracy()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.coeff = coeff
        
    def get_embedding(self,
                text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        result = self.encoder(embedded_text, mask)
        #print(result.shape)
        return result
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        
        
        batch_size = embedded_text.shape[0]
        num_tokens = embedded_text.shape[1]
        emb_size = embedded_text.shape[2]
        total_embed = []
        total_label = []
        total_binary_label = []
        #print(text, flush=True)
        #print(embedded_text, flush=True)
        #print(label, flush=True)
        
        if label is not None:
            for i in range(batch_size):
                label_ptr = 0
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        total_embed.append(embedded_text[i, j, :])
                        total_label.append(label[i][label_ptr])
                        total_binary_label.append(label[i][label_ptr] != label[i][label_ptr-1] if label_ptr > 0 else 0)
                        label_ptr += 1
        else:
            for i in range(batch_size):
                label_ptr = 0
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        total_embed.append(embedded_text[i, j, :])
                        label_ptr += 1
                    
        total_embed = torch.cat(total_embed).reshape((-1, emb_size))
        logits = self.classifier(total_embed)
        binary_logits = self.binary_classifier(total_embed)
        
        if label is not None:
            total_label = torch.Tensor(total_label).long().to(logits.device)
            total_binary_label = torch.Tensor(total_binary_label).long().to(logits.device)
        
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        binary_probs = torch.nn.functional.softmax(binary_logits, dim=-1)
        
        if label is not None:
            loss = self.coeff * torch.nn.functional.cross_entropy(logits, total_label) + (1-self.coeff) * torch.nn.functional.cross_entropy(binary_logits, total_binary_label)
            self.accuracy(logits, total_label)
            self.binary_accuracy(binary_logits, total_binary_label)
            return {'loss': loss, 'probs': probs, 'binary_probs': binary_probs}
        else:
            return {'probs': probs, 'binary_probs': binary_probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset), "binary_accuracy": self.binary_accuracy.get_metric(reset)}
    
    
class BinarySeq2SeqSegmenter(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 bos_id=0,
                 eos_id=2,
                 coeff=0.5):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.rnn = torch.nn.RNN(embedder.get_output_dim(), embedder.get_output_dim(), batch_first=True)
        self.classifier = torch.nn.Linear(embedder.get_output_dim(), num_labels)
        self.binary_classifier = torch.nn.Linear(embedder.get_output_dim(), 2)

        self.accuracy = CategoricalAccuracy()
        self.binary_accuracy = CategoricalAccuracy()

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.coeff = coeff
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        
        
        batch_size = embedded_text.shape[0]
        num_tokens = embedded_text.shape[1]
        emb_size = embedded_text.shape[2]
        total_embed = []
        total_label = []
        total_binary_label = []
        #print(text, flush=True)
        #print(embedded_text, flush=True)
        #print(label, flush=True)
        
        if label is not None:
            for i in range(batch_size):
                label_ptr = 0
                cur_embed = []
                
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        cur_embed.append(torch.unsqueeze(embedded_text[i, j, :], 0))
                        total_label.append(label[i][label_ptr])
                        total_binary_label.append(label[i][label_ptr] != label[i][label_ptr-1] if label_ptr > 0 else 0)
                        
                        label_ptr += 1
                cur_embed = torch.cat(cur_embed, dim=0)
                cur_embed = torch.unsqueeze(cur_embed, 0)
                
                output, h = self.rnn(cur_embed)
                total_embed.append(output[0, :])
            total_embed = torch.cat(total_embed, dim=0)
        else:
            for i in range(batch_size):
                label_ptr = 0
                cur_embed = []
                
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        cur_embed.append(torch.unsqueeze(embedded_text[i, j, :], 0))
                        label_ptr += 1
                cur_embed = torch.cat(cur_embed, dim=0)
                cur_embed = torch.unsqueeze(cur_embed, 0)
                    
                output, h = self.rnn(cur_embed)
                total_embed.append(output[0, :])
            total_embed = torch.cat(total_embed, dim=0)
                    
 
        logits = self.classifier(total_embed)
        binary_logits = self.binary_classifier(total_embed)
        binary_probs = torch.nn.functional.softmax(binary_logits, dim=-1)

        
        if label is not None:
            total_label = torch.Tensor(total_label).long().to(logits.device)
            total_binary_label = torch.Tensor(total_binary_label).long().to(logits.device)

        
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            loss = self.coeff * torch.nn.functional.cross_entropy(logits, total_label) + (1-self.coeff) * torch.nn.functional.cross_entropy(binary_logits, total_binary_label)
            self.accuracy(logits, total_label)
            self.binary_accuracy(binary_logits, total_binary_label)
            return {'loss': loss, 'probs': probs, 'binary_probs': binary_probs}
        else:
            return {'probs': probs, 'binary_probs': binary_probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset), "binary_accuracy": self.binary_accuracy.get_metric(reset)}
  

    
class BiheadSegmenter(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(embedder.get_output_dim() * 2, num_labels)
        self.accuracy = CategoricalAccuracy()
        self.bos_id = 0
        self.eos_id = 2
        
    def get_embedding(self,
                text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        result = self.encoder(embedded_text, mask)
        #print(result.shape)
        return result
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        
        
        batch_size = embedded_text.shape[0]
        num_tokens = embedded_text.shape[1]
        emb_size = embedded_text.shape[2]
        total_embed_bos = []
        total_embed_eos = []
        total_label = []
        #print(text, flush=True)
        #print(embedded_text, flush=True)
        #print(label, flush=True)
        
        if label is not None:
            for i in range(batch_size):
                label_ptr = 0
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        total_embed_eos.append(embedded_text[i, j, :])
                        total_label.append(label[i][label_ptr])
                        label_ptr += 1
                    elif text['bert_tokens']['token_ids'][i][j] == self.bos_id:
                        total_embed_bos.append(embedded_text[i, j, :])
                        
                if len(total_embed_bos) > len(total_embed_eos):
                    total_embed_bos = total_embed_bos[:len(total_embed_eos)]
                    
        else:
            for i in range(batch_size):
                label_ptr = 0
                for j in range(num_tokens):
                    if text['bert_tokens']['token_ids'][i][j] == self.eos_id:
                        total_embed_eos.append(embedded_text[i, j, :])
                        label_ptr += 1
                    elif text['bert_tokens']['token_ids'][i][j] == self.bos_id:
                        total_embed_bos.append(embedded_text[i, j, :])
                        
                if len(total_embed_bos) > len(total_embed_eos):
                    total_embed_bos = total_embed_bos[:len(total_embed_eos)]
                    
        total_embed_bos = torch.cat(total_embed_bos).reshape((-1, emb_size))
        total_embed_eos = torch.cat(total_embed_eos).reshape((-1, emb_size))
        
        #print(total_embed_bos.shape, flush=True)
        #print(total_embed_eos.shape, flush=True)
        
        total_embed = torch.cat([total_embed_bos, total_embed_eos], dim=-1)
        logits = self.classifier(total_embed)
        
        if label is not None:
            total_label = torch.Tensor(total_label).long().to(logits.device)
        
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, total_label)
            self.accuracy(logits, total_label)
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    
class SimpleMultiLabelClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.fbeta_measure = FBetaMeasure(average='macro')
        self.accuracy = Average()
        
    def get_embedding(self,
                text: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        result = self.encoder(embedded_text, mask)
        #print(result.shape)
        return result
    
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        #print(label.shape)
        probs = torch.nn.functional.sigmoid(logits)
        
        if label is not None:
            #print(label.shape, flush=True)
            loss = -torch.sum(torch.log(probs) * label + torch.log(1-probs) * (1-label)) / (probs.shape[1] * probs.shape[0])
            
            for j in range(probs.shape[0]):
                for i in range(probs.shape[1]):
                    self.accuracy(int((probs[j, i] >= 0.5).item() == label[j, i].item()))
            for i in range(probs.shape[1]):
                new_logits = torch.tensor((probs >= 0.5), dtype=torch.float32).to(probs.device)
                #new_logits[:, i] += 0.1
                self.fbeta_measure(new_logits, label[:, i])
                       
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        res = {"accuracy": self.accuracy.get_metric(reset), "precision": self.fbeta_measure.get_metric(reset)['precision']}
        #print(res, flush=True)
        return res

class AdversarialClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 simple_classifier: SimpleClassifier,
                 alpha: float = 0.05,
                 target: str = 'label',
                 freeze_topic: bool = False):
        super().__init__(vocab)
        self.simple_classifier = simple_classifier
        num_topics = vocab.get_vocab_size("topic_labels")
        self.topic_classifier = torch.nn.Linear(
            self.simple_classifier.encoder.get_output_dim(), 
            num_topics
        )
        self.alpha = alpha
        self.index_to_label = vocab.get_index_to_token_vocabulary('labels')
        self.label_to_dif_index = simple_classifier.vocab.get_token_to_index_vocabulary('labels')
        self.target = target
        self.freeze_topic = freeze_topic
        self.topic_accuracy = CategoricalAccuracy()
        
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None,
                topic: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        #print("label", label, flush=True)
        #print("topic", topic, flush=True)
        if self.freeze_topic:
            self.topic_classifier.requires_grad = False
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.simple_classifier.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.simple_classifier.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.simple_classifier.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        topic_logits = self.topic_classifier(encoded_text)
        
        if label is not None:
            label_device = label.get_device()
            mod_label = [self.label_to_dif_index[self.index_to_label[cur_label]] for cur_label in label.tolist()]
            mod_label = torch.tensor(mod_label, device=label_device, dtype=torch.long)
            self.simple_classifier.accuracy(logits, mod_label)
            
        if label is not None and topic is not None:
            if self.target == 'label':
                loss = torch.nn.functional.cross_entropy(logits, mod_label)
                loss -= self.alpha * torch.nn.functional.cross_entropy(topic_logits, topic)
            else:
                loss = torch.nn.functional.cross_entropy(topic_logits, topic)
            self.simple_classifier.accuracy(logits, mod_label)    
            self.topic_accuracy(topic_logits, topic)
            if self.target == 'label':
                return {'loss': loss, 'probs': probs}
            else:
                return {'loss': loss, 'probs': torch.nn.functional.softmax(topic_logits, dim=-1)}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.simple_classifier.accuracy.get_metric(reset), 
                "topic accuracy": self.topic_accuracy.get_metric(reset)}
    

def build_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BertPooler(transformer_model)
    return SimpleClassifier(vocab, embedder, encoder)

def build_segmentation_transformer_model(vocab: Vocabulary, transformer_model: str, eos_id=0, bos_id=2) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    
    return SimpleSegmenter(vocab, embedder, eos_id, bos_id)

def build_binary_segmentation_transformer_model(vocab: Vocabulary, transformer_model: str, eos_id=0, bos_id=2) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    
    return BinarySegmenter(vocab, embedder, eos_id, bos_id)

def build_binary_seq2seq_segmentation_transformer_model(vocab: Vocabulary, transformer_model: str, eos_id=0, bos_id=2) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    
    return BinarySeq2SeqSegmenter(vocab, embedder, eos_id, bos_id)

def build_seq2seq_segmentation_transformer_model(vocab: Vocabulary, transformer_model: str, eos_id=0, bos_id=2) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    
    return Seq2SeqSegmenter(vocab, embedder, eos_id, bos_id)

def build_bihead_segmentation_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
  
    return BiheadSegmenter(vocab, embedder)

def build_multilabel_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BertPooler(transformer_model)
    return SimpleMultiLabelClassifier(vocab, embedder, encoder)

def build_adversarial_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BertPooler(transformer_model)
    return SimpleClassifier(vocab, embedder, encoder)

def build_pool_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedder.get_output_dim(), averaged=True)
    #encoder = ClsPooler(embedding_dim=embedder.get_output_dim())
    return SimpleClassifier(vocab, embedder, encoder)

def build_multilabel_pool_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedder.get_output_dim(), averaged=True)
    #encoder = ClsPooler(embedding_dim=embedder.get_output_dim())
    return SimpleMultiLabelClassifier(vocab, embedder, encoder)

def build_elmo_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = ElmoTokenEmbedder()
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedder.get_output_dim(), averaged=True)
    
    return SimpleClassifier(vocab, embedder, encoder)

def build_simple_lstm_model(vocab: Vocabulary,
                            emb_size: int = 256,
                            hidden_size: int = 256,
                            num_layers: int = 2,
                            bidirectional: bool = True) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"bert_tokens": Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)}
    )
    encoder = LstmSeq2VecEncoder(
        input_size=emb_size, hidden_size=hidden_size, 
        num_layers=num_layers, bidirectional=bidirectional
    )
    return SimpleClassifier(vocab, embedder, encoder)

def build_simple_cnn_model(vocab: Vocabulary,
                           emb_size: int = 256,
                           output_dim: int = 256,
                           num_filters: int = 16,
                           ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5, 6)) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"bert_tokens": Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)}
    )
    encoder = CnnEncoder(
        embedding_dim=emb_size, ngram_filter_sizes=ngram_filter_sizes, output_dim=output_dim, 
        num_filters=num_filters,
    )
    return SimpleClassifier(vocab, embedder, encoder)