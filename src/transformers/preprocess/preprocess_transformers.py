import re

import pandas as pd
from natasha import (
    PER,
    Doc,
    MorphVocab,
    NamesExtractor,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)
from tqdm import tqdm

from src.constants import FeaturesConstants
from src.transformers.base import Transformer

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)


class CategoryFromTextPreprocess(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        preprocessed_text = []
        topics = []
        for i in input_data['title']:
            d = re.findall(r'\s[\w ]+,\xa0[\d\w, ]*\d\d:\d\d', i)
            if len(d) > 0:
                topic = (
                    d[0]
                    .split('\xa0')[0]
                    .split('\n')[-1]
                    .replace(',', '')
                    .lstrip()
                )
                i = i.replace(d[0], '\n')
            else:
                topic = None
            topics.append(topic)
            prepro = ' '.join(re.sub(r'[\n\xa0]', '', i).split())
            preprocessed_text.append(prepro)
        input_data['category_from_title'] = topics
        input_data['title_preprocessed'] = preprocessed_text
        return input_data


class AuthorsPreprocess(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        b = []
        for author in input_data['authors']:
            if author == '[]':
                b.append(None)
            else:
                b.append(re.findall(r'\w+', author))
        input_data['authors'] = b
        return input_data


class TagsPreprocess(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        c = []
        for author in input_data['tags']:
            if author == '[]':
                c.append(None)
            else:
                c.append(re.findall(r'\w+', author))
        input_data['tags'] = c
        return input_data


class FeaturePreprocess(Transformer):
    def transform(
        self, input_data: pd.DataFrame, mode: str = 'train'
    ) -> pd.DataFrame:
        identifier = ['document_id', 'page_id', 'session']
        int_features = ['ctr']
        date_features = ['publish_date']
        title = [
            'title',
            'news_title',
            'news_header_yandex',
            'title_preprocessed',
        ]
        authors = ['authors', 'news_authors']
        tags = ['tags', 'news_tags']
        category = ['category', 'news_topic', 'category_from_title']
        text = ['news_text', 'news_text_overview']
        text_features = ['news_amount_of_paragraphs']
        between_text_features = [
            'news_amount_of_inline_items',
            'news_inline_titles',
        ]
        image = ['news_has_image', 'news_image_title']
        others = [
            'keywords_parsed',
            'twitter_card_parsed',
            'twitter_image_parsed',
            'vk_image_parsed',
            'yandex_recommendation_image_parsed',
            'yandex_recommendation_category_parsed',
        ]
        output = input_data[
            identifier
            + int_features
            + date_features
            + title
            + authors
            + tags
            + category
            + text
            + text_features
            + between_text_features
            + image
            + others
        ]
        if mode == 'train':
            output = pd.concat(
                (output, input_data[FeaturesConstants.target]), axis=1
            )
        output = output.rename(
            {
                'news_title': 'title_parsed',
                'news_header_yandex': 'title_parsed_from_yandex',
                'news_topic': 'category_parsed',
                'news_authors': 'authors_parsed',
                'news_tags': 'tags_parsed',
                'news_text': 'news_text_parsed',
                'news_text_overview': 'news_text_overview_parsed',
                'news_amount_of_paragraphs': 'news_amount_of_paragraphs_parsed',
                'news_amount_of_inline_items': 'news_amount_of_inline_items_parsed',
                'news_inline_titles': 'news_inline_titles_parsed',
                'news_has_image': 'news_has_image_parsed',
                'news_image_title': 'news_image_title_parsed',
            },
            axis=1,
        )
        output['authors_parsed'] = output['authors_parsed'].apply(
            lambda x: [' '.join(m.split()) for m in x]
        )
        output['authors_parsed'] = output['authors_parsed'].apply(
            lambda x: x if x != [] else None
        )
        output['tags_parsed'] = output['tags_parsed'].apply(
            lambda x: x if x != [] else None
        )
        output['news_inline_titles_parsed'] = output[
            'news_inline_titles_parsed'
        ].apply(lambda x: x if x != [] else None)
        output['ctr'] = output['ctr'].apply(lambda x: x if x != 0 else None)
        output['news_image_title_parsed'] = output[
            'news_image_title_parsed'
        ].apply(lambda x: x if x != '' else None)
        #         data['lemmatized'] = data['prepro_text'].apply(lambda x: ' '.join([lru_lemmatize(p) if p.islower() else p for p in x.split()]))
        return output


def natasha_result(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        span.normalize(morph_vocab)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    for span in doc.spans:
        if span.type == PER:
            span.extract_fact(names_extractor)
    lemmatized = [_.lemma for _ in doc.tokens]
    if doc.spans is []:
        return None, None, None, None
    else:
        per = [i.normal for i in doc.spans if i.type == 'PER']
        loc = [i.normal for i in doc.spans if i.type == 'LOC']
        org = [i.normal for i in doc.spans if i.type == 'ORG']
    return (
        per if per != [] else None,
        loc if loc != [] else None,
        org if org != [] else None,
        lemmatized if lemmatized != [] else None,
    )


class NatashaTransformer(Transformer):
    def transform(
        self, input_data: pd.DataFrame, mode: str = 'train'
    ) -> pd.DataFrame:
        v1 = []
        v2 = []
        v3 = []
        v4 = []
        for i in tqdm(input_data['title_preprocessed'], desc=mode):
            a, b, c, d = natasha_result(i)
            v1.append(a)
            v2.append(b)
            v3.append(c)
            v4.append(d)
        input_data['PER_TITLE'] = v1
        input_data['ORG_TITLE'] = v2
        input_data['LOC_TITLE'] = v3
        input_data['news_title_lemmatized'] = v4
        #
        # v1 = []
        # v2 = []
        # v3 = []
        # v4 = []
        # for i in tqdm(input_data['news_text_parsed'], desc=mode):
        #     a, b, c, d = natasha_result(i)
        #     v1.append(a)
        #     v2.append(b)
        #     v3.append(c)
        #     v4.append(d)
        # input_data['PER_NEWS_TEXT'] = v1
        # input_data['ORG_NEWS_TEXT'] = v2
        # input_data['LOC_NEWS_TEXT'] = v3
        # input_data['news_text_parsed_lemmatized'] = v4
        return input_data
