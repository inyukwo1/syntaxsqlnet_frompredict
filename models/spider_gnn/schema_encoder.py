import torch
import torch.nn as nn
from models.spider_gnn.spider_db_context import SpiderDBContext
from models.spider_gnn.spider_knowledgegraph import SpiderKnowledgeGraphField
from models.spider_gnn.spider_world import SpiderWorld
from models.spider_gnn.spider_utils import fix_number_value, disambiguate_items
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import WordTokenizer
from spacy.symbols import ORTH, LEMMA
from allennlp.modules import TextFieldEmbedder


class SchemaEncoder(nn.Module):
    def __init__(self):
        super(SchemaEncoder, self).__init__()
        spacy_tokenizer = SpacyWordSplitter(pos_tags=True)
        spacy_tokenizer.spacy.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
        self._tokenizer = WordTokenizer(spacy_tokenizer)
        pass

    def forward(self, datum):
        db_id = datum["db_id"]
        question_token = datum["question"]
        query_tokens = None
        if 'query_toks' in datum:
            # we only have 'query_toks' in example for training/dev sets

            # fix for examples: we want to use the 'query_toks_no_value' field of the example which anonymizes
            # values. However, it also anonymizes numbers (e.g. LIMIT 3 -> LIMIT 'value', which is not good
            # since the official evaluator does expect a number and not a value
            ex = fix_number_value(datum)

            # we want the query tokens to be non-ambiguous (i.e. know for each column the table it belongs to,
            # and for each table alias its explicit name)
            # we thus remove all aliases and make changes such as:
            # 'name' -> 'singer@name',
            # 'singer AS T1' -> 'singer',
            # 'T1.name' -> 'singer@name'
            try:
                query_tokens = disambiguate_items(datum['db_id'], datum['query_toks_no_value'],
                                                  self._tables_file, allow_aliases=False)
            except Exception as e:
                # there are two examples in the train set that are wrongly formatted, skip them
                print(f"error with {datum['query']}")
                print(e)

        db_context = SpiderDBContext(db_id, question_token, tokenizer=self._tokenizer,
                                     tables_file=self._tables_file, dataset_path=self._dataset_path)
        table_field = SpiderKnowledgeGraphField(db_context.knowledge_graph,
                                                db_context.tokenized_utterance,
                                                self._utterance_token_indexers,
                                                entity_tokens=db_context.entity_tokens,
                                                include_in_vocab=False,  # TODO: self._use_table_for_vocab,
                                                max_table_tokens=None)  # self._max_table_tokens)

        utterance = TextField(db_context.tokenized_utterance, {'tokens': SingleIdTokenIndexer()}).as_tensor()
        world = MetadataField(SpiderWorld(db_context, query_tokens)).as_tensor()
        schema = table_field.as_tensor()

        schema_text = schema['text']


