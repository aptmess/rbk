import scrapy as sp


class TopicItem(sp.Item):
    forum_id: int = sp.Field()
    topic_id: int = sp.Field()
    topic_name: str = sp.Field()
    topic_url: str = sp.Field()


class PostItem(sp.Item):
    post_id = sp.Field()
    post_url = sp.Field()
    post_likers = sp.Field()
    post_text = sp.Field()
    post_created_at = sp.Field()
    post_rating = sp.Field()
    is_deleted = sp.Field()
    related_to = sp.Field()
    topic_id = sp.Field()
    author_id = sp.Field()


class AuthorItem(sp.Item):
    author_id = sp.Field()


class NewsItem(sp.Item):
    document_id = sp.Field()
    news_title = sp.Field()
    news_topic = sp.Field()
    news_header_yandex = sp.Field()
    news_text_overview = sp.Field()
    news_text = sp.Field()
    news_amount_of_paragraphs = sp.Field()
    news_has_image = sp.Field()
    news_image_title = sp.Field()
    news_amount_of_inline_items = sp.Field()
    news_inline_titles = sp.Field()
    news_authors = sp.Field()
    news_tags = sp.Field()
    keywords_parsed = sp.Field()
    twitter_card_parsed = sp.Field()
    twitter_image_parsed = sp.Field()
    vk_image_parsed = sp.Field()
    yandex_recommendation_image_parsed = sp.Field()
    yandex_recommendation_category_parsed = sp.Field()
