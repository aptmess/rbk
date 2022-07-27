DIV = "div[@class='mfd-post-top']"
DIV_0 = DIV + "/div[@class='mfd-post-top-0']"
DIV_1 = DIV + "/div[@class='mfd-post-top-1']"
DIV_2 = DIV + "/div[@class='mfd-post-top-2']"
DIV_0_LINK = DIV_0 + "/a[@class='mfd-poster-link']"
DIV_0_ANONYMOUS = DIV_0 + "/a[@class='mfd-anonymous-link']"


class POSTLinks:
    post_id = DIV_0 + '/@id'
    is_post_deleted = 'table/' + "/div[@class='mfd-post-remark']" + '/text()'
    created_at = DIV_1 + "/a[@class='mfd-post-link']" + '/text()'
    post_rating = DIV_2 + '/span' + '/text()'
    post_likers = DIV_2 + "/div[@class='mfd-post-ratingdetails']/table/tr"
    answered_posts_ids_path = (
        'table/'
        + "/div[@class='mfd-post-text']"
        + '/blockquote'
        + "/div[@class='mfd-quote-info']"
        + '/a[2]'
        + '/@href'
    )
    post_text = (
        'table/'
        + "/div[@class='mfd-post-text']"
        + "/div[@class='mfd-quote-text']"
    )


class AUTHORLinks:
    author_id = DIV_0_LINK + '/@title'
    author_name = DIV_0_LINK


class AnonymousAuthorLinks:
    author_id = DIV_0_ANONYMOUS + '/@title'
    author_name = DIV_0_ANONYMOUS + '/text()'


news_info_url = (
    "//div[@class='mfd-body-container']/"
    "div[@class='mfd-body-container-inner']"
    "/table/tr/td[@class='mfd-content-container']"
    "/table/tr/td[@class='mfd-column-left']"
)

NEWS_PATH = (
    "//body[@class='news']"
    "/div[@class='l-window l-window-overflow-mob']"
    "/div[@class='g-relative g-clear']"
    "/div[@class='l-col-container']"
    "/div[@class='l-table']"
    "/div[@class='js-rbcslider']"
    "/div[@class='js-rbcslider-slide rbcslider__slide ']"
    "/div[@class='article g-relative js-rbcslider-article ']"
    "/div[@class='l-table']"
    "/div[@class='l-col-main']"
    "/div[@class='l-col-center-590 article__content']"
)

HEADER = NEWS_PATH + "/div[@class='article__header js-article-header']"
TEXT = NEWS_PATH + "/div[@class='article__text article__text_free']"
TAGS_AUTHORS = (
    NEWS_PATH
    + "/div[@class='article__tabs-wrapper js-article-tabs-wrapper']/div[@class='article__tabs__panes']"
)


class NewsLinks:
    TITLE_HEADER = HEADER + "/div[@class='article__header__title']/h1/text()"
    TOPIC_HEADER = (
        HEADER + "/div[@class='article__header__info-block']/a/text()"
    )
    YANDEX_HEADER = HEADER + "/div[@class='article__header__yandex']/text()"
    TEXT_OVERVIEW = TEXT + "/div[@class='article__text__overview']/span/text()"
    TEXT_NEWS = TEXT + '/p'
    NEWS_HAS_IMAGE = TEXT + "/div[@class='article__main-image']"
    NEWS_IMAGE_TITLE = (
        NEWS_HAS_IMAGE + "/div[@class='article__main-image__title']/span/text()"
    )
    NEWS_AMOUNT_OF_INLINE_ITEMS = TEXT + "/div[@class='article__inline-item']"
    NEWS_INLINE_TITLES = (
        NEWS_AMOUNT_OF_INLINE_ITEMS
        + "/div[@class='article__inline-item__wrap']"
        "/div[@class='article__inline-item__content']"
        '/a/span/text()'
    )
    AUTHORS = (
        TAGS_AUTHORS + "/div[@data-ptab-id='article-tab-authors']/"
        "div[@class='article__authors']/"
        "div[@class='article__authors__row']/"
        "div[@class='article__authors__author__wrap']"
        "/a/span[@class='article__authors__author__name']/text()"
    )
    TAGS = (
        TAGS_AUTHORS + "/div[@data-ptab-id='article-tab-tags']/"
        "div[@class='article__tags']/"
        "div[@class='article__tags__container']/"
        '/a/text()'
    )
    # AUTHORS = TAGS_AUTHORS + ""
    # news_title = news_info_url + '/table/tr/td/h1/text()'
    # news_text = news_info_url + "/div[@class='m-content']/p/text()"
    # news_tags = news_info_url + "/div[@class='mfd-related-companies']"
    # news_source = news_info_url + "/div[@class='m-content']/p/a/text()"
    # news_create_at = (
    #     news_info_url + '/table/'
    #     'tr'
    #     "/td[@class='mfd-content-datetime']"
    #     "/span[@class='mfd-content-date']/text()"
    # )
