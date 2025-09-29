from review_clusterer import ReviewClusterer
from llm import LMStudioClient

llm_client = LMStudioClient(
    api_url="http://localhost:1234/v1/chat/completions",
    model="qwen3-4b-2507",
    temperature=0.1
)

clusterer = ReviewClusterer(
    embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
    llm_client=llm_client
)

texts = clusterer.load_reviews("data/reviews_banki_ru.json", max_reviews=250)

if texts:
    topics, cluster_info = clusterer.fit_clusters(texts, min_topic_size=5, use_llm=True)

    clusterer.save_cluster_names("data/all_topics.json")
    
    topic_info = clusterer.get_topic_info()
else:
    print("Не удалось загрузить данные для кластеризации")