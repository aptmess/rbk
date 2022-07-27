import typing as t

from scrapy.http import Response
from scrapy.utils.python import to_unicode
from six.moves.urllib.parse import urljoin

from scrapper.items import NewsItem
from scrapper.paths import NewsLinks
from scrapper.services.pipeline import Compose
from scrapper.services.transformers.news_transformers import (
    NewsAmountOfInlineItems,
    NewsAmountOfParagraphs,
    NewsAuthorsTags,
    NewsHasImage,
    NewsImageTitle,
    NewsInlineTitles,
    NewsText,
    NewsTitle,
    NewsTitleYandexHeader,
)


def parse_one_new(response: Response, base_url: str) -> t.Iterator[t.Any]:
    if 300 <= response.status < 400:
        location = to_unicode(response.headers['location'].decode('latin1'))
        request = response.request
        redirected_url = urljoin(request.url, location)

        if response.status in (301, 307) or request.method == 'HEAD':
            redirected = request.replace(url=redirected_url)
            yield redirected
        else:
            redirected = request.replace(
                url=redirected_url, method='GET', body=''
            )
            redirected.headers.pop('Content-Type', None)
            redirected.headers.pop('Content-Length', None)
            yield redirected
    news_item = NewsItem()
    news_compose = Compose(
        [
            NewsTitle(NewsLinks.TITLE_HEADER, 'news_title', response),
            NewsTitle(NewsLinks.TOPIC_HEADER, 'news_topic', response),
            NewsTitleYandexHeader(
                NewsLinks.YANDEX_HEADER, 'news_header_yandex', response
            ),
            NewsTitleYandexHeader(
                NewsLinks.TEXT_OVERVIEW, 'news_text_overview', response
            ),
            NewsInlineTitles(
                NewsLinks.NEWS_INLINE_TITLES,
                'news_inline_titles',
                response,
                get_all=True,
            ),
            NewsText(NewsLinks.TEXT_NEWS, 'news_text', response, get_all=True),
            NewsAmountOfParagraphs(
                NewsLinks.TEXT_NEWS,
                'news_amount_of_paragraphs',
                response,
                get_all=True,
            ),
            NewsHasImage(
                NewsLinks.NEWS_HAS_IMAGE,
                'news_has_image',
                response,
                get_all=True,
            ),
            NewsAmountOfInlineItems(
                NewsLinks.NEWS_AMOUNT_OF_INLINE_ITEMS,
                'news_amount_of_inline_items',
                response,
                get_all=True,
            ),
            NewsAuthorsTags(
                NewsLinks.AUTHORS, 'news_authors', response, get_all=True
            ),
            NewsAuthorsTags(
                NewsLinks.TAGS, 'news_tags', response, get_all=True
            ),
            NewsImageTitle(
                NewsLinks.NEWS_IMAGE_TITLE, 'news_image_title', response
            ),
        ]
    )
    news, result = news_compose(news_item)
    if result:
        news['document_id'] = response.meta['document_id']
        news['keywords_parsed'] = (
            response.xpath("//head/meta[@name='keywords']/@content")
            .get()
            .split(', ')
        )
        news['twitter_card_parsed'] = response.xpath(
            "//head/meta[@name='twitter:card']/@content"
        ).get()
        news['twitter_image_parsed'] = response.xpath(
            "//head/meta[@name='twitter:image']/@content"
        ).get()
        news['vk_image_parsed'] = response.xpath(
            "//head/meta[@name='vk:image']/@content"
        ).get()
        news['yandex_recommendation_image_parsed'] = response.xpath(
            "//head/meta[@property='yandex_recommendations_image']/@content"
        ).get()
        news['yandex_recommendation_category_parsed'] = response.xpath(
            "//head/meta[@property='yandex_recommendations_category']/@content"
        ).get()

        yield news
