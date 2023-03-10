import allennlp
from allennlp.data.token_indexers import (
    TokenIndexer, PretrainedTransformerIndexer, SingleIdTokenIndexer,
    ELMoTokenCharactersIndexer,
)
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer, WhitespaceTokenizer, SpacyTokenizer
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField, ListField
import tempfile
import torch
from typing import Dict, Iterable, Tuple
import pandas as pd
from DeBERTa import deberta
from nltk.tokenize.punkt import PunktSentenceTokenizer
from allennlp.data.fields.multilabel_field import MultiLabelField


class ClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 lower: bool = False,
                 top_tokenizer=None):
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.lower = lower
        self.top_tokenizer = top_tokenizer
        
    def text_to_instance(self, string: str, label: str = None) -> Instance:
        tmp_string = string.lower() if self.lower else string
        if self.top_tokenizer is not None:
            tmp_string = ' '.join(map(str, self.top_tokenizer.tokenize(tmp_string))).replace('##', '')
            
        tokens = self.tokenizer.tokenize(tmp_string)[:self.max_tokens]
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"text": sentence_field}
        if label is not None:
            fields["label"] = LabelField(str(label))
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        for text, label in zip(dataset_df['text'], dataset_df['target']):
            if type(text) == str:
                yield self.text_to_instance(text, str(label))
                
class SegmentationDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 lower: bool = False):
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.lower = lower
        
    def text_to_instance(self, string: str, label: str = None) -> Instance:
        tmp_string = string.lower() if self.lower else string
        sentences = tmp_string.split('[sent_sep]')
            
        tokens = []
        
        for sentence in sentences:
            tokens += self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_tokens]
        
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"text": sentence_field}
        
        if label is not None:
            fields["label"] = ListField([LabelField(genre_label) for genre_label in label.split()])
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        for text, label in zip(dataset_df['text'], dataset_df['target']):
            if type(text) == str:
                yield self.text_to_instance(text, label)
                
                
class MultiLabelClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 lower: bool = False,
                 top_tokenizer=None):
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.lower = lower
        self.top_tokenizer = top_tokenizer
        
    def text_to_instance(self, string: str, label: str = None) -> Instance:
        tmp_string = string.lower() if self.lower else string
        if self.top_tokenizer is not None:
            tmp_string = ' '.join(map(str, self.top_tokenizer.tokenize(tmp_string))).replace('##', '')
            
        tokens = self.tokenizer.tokenize(tmp_string)[:self.max_tokens]
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"text": sentence_field}
        if label is not None:
            label = str(label)
            fields["label"] = MultiLabelField(label.split(','))
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        for text, label in zip(dataset_df['text'], dataset_df['target']):
            if type(text) == str:
                yield self.text_to_instance(text, label)
                
        
class DebertaDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 max_tokens: int = None,
                 lower: bool = False):
        super().__init__(lazy)
        self.tokenizer = deberta.GPT2Tokenizer()
        self.max_tokens = max_tokens
        self.lower = lower
        
    def text_to_instance(self, string: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(string.lower() if self.lower else string)
        fields = {"text": tokens}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        for text, label in zip(dataset_df['text'], dataset_df['target']):
            if type(text) == str:
                yield self.text_to_instance(text[:5000], label)
        
class DomainDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 lower: bool = False):
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.lower = lower
        
    def text_to_instance(self, string: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(string.lower() if self.lower else string)
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"text": sentence_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        for text, label in zip(dataset_df['text'], dataset_df['domain']):
            yield self.text_to_instance(text, label)
            
          
class AdversarialDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 lower: bool = False):
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.lower = lower
        
    def text_to_instance(self, string: str, label: str = None, topic: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(string.lower() if self.lower else string)
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"text": sentence_field}
        if label is not None:
            fields["label"] = LabelField(label)
        if topic is not None:
            fields["topic"] = LabelField(topic, label_namespace='topic_labels')
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        if 'target' not in dataset_df:
            dataset_df['target'] = ['W' for _ in range(dataset_df.shape[0])]
        for text, label, topic in zip(dataset_df['text'], dataset_df['target'], dataset_df['topic']):
            yield self.text_to_instance(text[:5000], label, str(topic))
        
        
class SmartClassificationDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 lower: bool = False):
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.lower = lower
        
    def text_to_instance(self, string: str, label: str) -> Instance:
        tokens = self.tokenizer.tokenize(string.lower() if self.lower else string)
        
        for first_token_id in range(1, len(tokens) - 1, self.max_tokens - 2):
            last_token_id = min(first_token_id + self.max_tokens - 2, len(tokens) - 1)
            sentence_field = TextField([tokens[0]] + tokens[first_token_id:last_token_id] + [tokens[-1]], self.token_indexers)
            yield Instance({"text": sentence_field, "label": LabelField(label)})

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset_df = pd.read_csv(file_path)
        for text, label in zip(dataset_df['text'], dataset_df['target']):
            yield from self.text_to_instance(text, label)
            

def read_data(train_path: str, val_path: str, train_reader: DatasetReader,
              val_reader: DatasetReader = None) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    print(type(train_reader), train_path)
    training_data = train_reader.read(train_path)
    if val_reader is None:
        validation_data = train_reader.read(val_path)
    else:
        validation_data = val_reader.read(val_path)
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)

def build_transformer_dataset_reader(transformer_model, MAX_TOKENS=512, lower=False, top_tokenizer=None) -> DatasetReader:
    #transformer_model = 'bert-base-cased'
    tokenizer = PretrainedTransformerTokenizer(transformer_model, max_length=MAX_TOKENS-2)
    token_indexers = {'bert_tokens': PretrainedTransformerIndexer(transformer_model)}
    return ClassificationDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=MAX_TOKENS, lower=lower, top_tokenizer=top_tokenizer
    )

def build_segmentation_dataset_reader(transformer_model, MAX_TOKENS=512, lower=False) -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(transformer_model, max_length=MAX_TOKENS-2)
    token_indexers = {'bert_tokens': PretrainedTransformerIndexer(transformer_model)}
    
    return SegmentationDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=MAX_TOKENS, lower=lower
    )

def build_multilabel_transformer_dataset_reader(transformer_model, MAX_TOKENS=512, lower=False, top_tokenizer=None) -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(transformer_model, max_length=MAX_TOKENS-2)
    token_indexers = {'bert_tokens': PretrainedTransformerIndexer(transformer_model)}
    return MultiLabelClassificationDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=MAX_TOKENS, lower=lower, top_tokenizer=top_tokenizer
    )

def build_elmo_dataset_reader(lower=False) -> DatasetReader:
    tokenizer = WhitespaceTokenizer()
    token_indexers = {'bert_tokens': ELMoTokenCharactersIndexer()}
    return ClassificationDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=300, lower=lower
    )

def build_smart_transformer_dataset_reader(transformer_model, MAX_TOKENS=512, lower=False) -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(transformer_model, max_length=None)
    token_indexers = {'bert_tokens': PretrainedTransformerIndexer(transformer_model)}
    return SmartClassificationDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=MAX_TOKENS, lower=lower
    )

def build_domain_dataset_reader(lower=False) -> DatasetReader:
    tokenizer = WhitespaceTokenizer()
    token_indexers = {'tokens': SingleIdTokenIndexer()}
    return DomainDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=None, lower=lower
    )

def build_adversarial_dataset_reader(transformer_model, MAX_TOKENS=512, lower=False) -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(transformer_model, max_length=MAX_TOKENS-2)
    token_indexers = {'bert_tokens': PretrainedTransformerIndexer(transformer_model)}
    return AdversarialDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=MAX_TOKENS, lower=lower
    )

def build_dataset_reader(transformer_model=None, lower=False) -> DatasetReader:
    if transformer_model is not None:
        tokenizer = PretrainedTransformerTokenizer(transformer_model)
    else:
        tokenizer = WhitespaceTokenizer()
    token_indexers = {'bert_tokens': SingleIdTokenIndexer()}
    return ClassificationDatasetReader(
        tokenizer=tokenizer, token_indexers=token_indexers,
        max_tokens=None, lower=lower
    )

# The other `build_*` methods are things we've seen before, so they are
# in the setup section above.
def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader