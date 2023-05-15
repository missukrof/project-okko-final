import numpy as np
import pandas as pd

import re
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from typing import List

import sys
sys.path.append('C:/Users/a.kuznetsova/Documents/Python Scripts/okko/project-okko-team-work-final/project-okko-final')

from configs.config import settings
from utils.utils import (
    read_parquet_from_local_path,
    dump_parquet_to_local_path
    )

from fire import Fire

def word_tokenize_clean(doc: str, stop_words: list):
    '''
    tokenize from string to list of words
    '''
    # init lemmatizer to avoid slow performance
    mystem = Mystem()

    # split into lower case word tokens \w lemmatization
    tokens = list(set(mystem.lemmatize(doc.lower())))
  
    # remove tokens that are not alphabetic (including punctuation) and not a stop word
    tokens = [word for word in tokens if word.isalpha() and not word in stop_words \
              not in list(punctuation)]
    return tokens


def get_clean_tags_array(agg_tags: pd.DataFrame,
                         text_col = 'tag'):
    '''text preprocessing
    '''
    tags_corpus = agg_tags[text_col].values
    tags_corpus = [re.sub('-[!/()0-9]', '', x) for x in tags_corpus]
    stop_words = stopwords.words('russian')

    # preprocess corpus of movie tags before feeding it into Doc2Vec model
    tags_doc = [TaggedDocument(words = word_tokenize_clean(D, stop_words), tags = [str(i)]) for i, D in enumerate(tags_corpus)]

    return tags_doc


def convert_to_string(df, col):
    df[col] = df[col].apply(lambda x: ', '.join(x))
    return df


def train_embeddings(tags_doc: np.array,
                     epochs: int = 20,
                     vec_size: int = 50,
                     alpha: float = .02,
                     min_alpha: float =  0.00025,
                     min_count: int = 5,
                     save_path: str = None):
    """
    fit doc2vec model to prepared corpus
    :tags_doc: result of get_clean_tags_array()
    :max_epocs: int
    :vec_size: int
    :alpha: float
    """
    #initialize
    model = Doc2Vec(vector_size = vec_size,
                    alpha = alpha, 
                    min_alpha = min_alpha,
                    min_count = min_count,
                    dm = 0)
    
    #generate vocab from all tag docs
    model.build_vocab(tags_doc)
    
    #train model
    model.train(tags_doc,
                total_examples = model.corpus_count,
                epochs = epochs)
    
    #save model to dir
    if save_path:
        model.save(f'{save_path}/d2v_model.pkl')
    
    return model


def generate_embeddings(df, id_col, title_col, merge_cols: List[str], list_cols: List[str] = None, name_cols: List[str] = None):
    sample = df.copy()
    sample = sample.fillna(' ')

    if list_cols is not None and name_cols is not None:
        list_name_cols = list(set(list_cols) & set(name_cols))
        ord_name_cols = list(set(name_cols) - (set(list_cols) & set(name_cols)))

        if len(list_name_cols) > 0:
            for i in list_name_cols:
                sample[i] = sample[i].apply(lambda x: [a.replace(' ', '') for a in x])
        elif len(ord_name_cols) > 0:
            for i in ord_name_cols:
                sample[i] = sample[i].apply(lambda x: x.replace(' ', ''))
    elif list_cols is None and name_cols is not None:
        for i in name_cols:
            sample[i] = sample[i].apply(lambda x: x.replace(' ', ''))

    if list_cols is not None:
        for i in list_cols:
          sample = convert_to_string(sample, i)
  
    sample['merged_columns'] = sample[merge_cols].apply(lambda x: '. '.join(x.dropna().astype(str)), axis=1)
    # define model_index and make it as string
    sample = sample.reset_index().rename(columns = {'index': 'model_index'})
    sample['model_index'] = sample['model_index'].astype(str)
    name_mapper = dict(zip(sample['model_index'].astype(int), sample[title_col].str.lower()))

    # let's check what do we have
    ## tag = movie index
    tags_doc = get_clean_tags_array(sample, 'merged_columns')
    model = train_embeddings(tags_doc, vec_size=1)

    # load trained embeddings 
    movies_vectors = model.dv.vectors

    return movies_vectors


def add_embeddings_to_dataframe():

    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    movies_metadata['movie_meta_emb'] = generate_embeddings(
        df=movies_metadata,
        id_col=settings.MOVIES_FEATURES.MOVIE_IDS,
        title_col='title',
        merge_cols=['title', 'entity_type', 'genres', 'country'],
        list_cols=['genres', 'country']
        )

    movies_metadata['movie_cast_emb'] = generate_embeddings(
        df=movies_metadata,
        id_col=settings.MOVIES_FEATURES.MOVIE_IDS,
        title_col='title',
        merge_cols=['actors', 'director'],
        list_cols=['actors', 'director'],
        name_cols=['actors', 'director']
        )

    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

if __name__ == '__main__':
    Fire(
    {
        'add_embeddings_to_dataframe': add_embeddings_to_dataframe
        }
    )