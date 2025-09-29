# review_clusterer.py
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import re

# Кластеризация
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class ReviewClusterer:
    def __init__(self, 
                 embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 llm_client = None):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.topic_model = None
        self.cluster_descriptions = {}
        
        self.llm_client = llm_client
        
        try:
            nltk.download('stopwords', quiet=True)
            self.russian_stopwords = stopwords.words('russian')
        except:
            self.russian_stopwords = []
            logger.warning("Не удалось загрузить стоп-слова NLTK")
    
    def load_reviews(self, filepath: str, max_reviews: Optional[int] = None) -> List[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            for item in data:
                if isinstance(item, dict) and 'text' in item and item['text']:
                    texts.append(item['text'])
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())
                
                if max_reviews and len(texts) >= max_reviews:
                    break
            
            logger.info(f"Загружено {len(texts)} отзывов из файла {filepath}")
            return texts
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных из {filepath}: {e}")
            return []
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                continue
                
            text = text.lower()
            text = re.sub(r'[^а-яёa-z0-9\s\.\,\!\?]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text:
                processed_texts.append(text)
        
        return processed_texts
    
    def fit_clusters(self, texts: List[str], min_topic_size: int = 10, use_llm: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
        if not texts:
            logger.error("Нет текстов для кластеризации")
            return np.array([]), {}
        
        logger.info(f"Запуск кластеризации для {len(texts)} отзывов...")
        
        processed_texts = self.preprocess_texts(texts)
        
        umap_model = UMAP(
            n_neighbors=min(15, len(processed_texts) // 10),
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        vectorizer_model = CountVectorizer(
            stop_words=self.russian_stopwords,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="russian",
            calculate_probabilities=False,
            verbose=True,
            **kwargs
        )
        
        topics, _ = self.topic_model.fit_transform(processed_texts)

        if isinstance(topics[0], (list, np.ndarray)):
            topics = np.array([t[0] if len(t) > 0 else -1 for t in topics])
        else:
            topics = np.array(topics)
        
        self._generate_cluster_descriptions(texts, topics, use_llm)
        
        logger.info(f"Кластеризация завершена. Выявлено {len(set(topics)) - (1 if -1 in topics else 0)} тем")
        
        return topics, self.cluster_descriptions
    
    def _generate_cluster_descriptions(self, texts: List[str], topics: np.ndarray, use_llm: bool = True) -> None:
        if self.topic_model is None:
            return
        
        topic_info = self.topic_model.get_topic_info()
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            
            if topic_id == -1:
                self.cluster_descriptions[topic_id] = {
                    "name": "Шум/Прочее",
                    "description": "Неразмеченные отзывы, не относящиеся к основным темам",
                    "size": row['Count'],
                    "keywords": ["разное", "шум"]
                }
            else:
                topic_words = self.topic_model.get_topic(topic_id)
                keywords = [word for word, _ in topic_words[:10]]
                
                topic_texts = np.array(texts)[topics == topic_id].tolist()
                sample_texts = topic_texts[:5]
                
                if use_llm and len(sample_texts) > 0 and self.llm_client:
                    llm_result = self.llm_client.generate_topic_name_and_description(keywords, sample_texts)
                    name = llm_result.get("name", "")
                    description = llm_result.get("description", "")
                    
                    if not name or not description:
                        name = self._create_topic_name_fallback(keywords)
                        description = self._create_topic_description_fallback(keywords)
                else:
                    name = self._create_topic_name_fallback(keywords)
                    description = self._create_topic_description_fallback(keywords)
                
                self.cluster_descriptions[topic_id] = {
                    "name": name,
                    "description": description,
                    "size": row['Count'],
                    "keywords": keywords[:5],
                    "full_keywords": keywords,
                    "sample_texts": sample_texts[:3]
                }
    
    def _create_topic_name_fallback(self, keywords: List[str]) -> str:
        banking_terms = {
            'карт': 'Банковские карты',
            'кредит': 'Кредитные продукты', 
            'вклад': 'Вклады и депозиты',
            'ипотек': 'Ипотечное кредитование',
            'приложен': 'Мобильное приложение',
            'обслуж': 'Клиентское обслуживание',
            'перевод': 'Денежные переводы',
            'платеж': 'Платежные операции',
            'страхов': 'Страховые продукты',
            'инвест': 'Инвестиционные продукты',
            'онлайн': 'Онлайн-банкинг',
            'отдел': 'Обслуживание в отделениях',
            'счет': 'Банковские счета',
            'деньг': 'Денежные операции',
            'банкомат': 'Банкоматы',
            'кэшбэк': 'Кэшбэк и бонусы',
            'процент': 'Процентные ставки',
            'заявк': 'Оформление заявок',
            'одобрен': 'Одобрение продуктов',
            'отказ': 'Отказы в обслуживании'
        }
        
        for keyword in keywords:
            for term, name in banking_terms.items():
                if term in keyword.lower():
                    return name
        
        return f"Тема: {keywords[0] if keywords else 'Разное'}"
    
    def _create_topic_description_fallback(self, keywords: List[str]) -> str:
        if not keywords:
            return "Общая банковская тема"
        
        main_keywords = keywords[:3]
        return f"Обсуждаются вопросы, связанные с {', '.join(main_keywords)}"
    
    def get_topic_info(self) -> pd.DataFrame:
        if self.topic_model is None:
            return pd.DataFrame()
        
        topic_info = self.topic_model.get_topic_info()
        
        topic_info['custom_name'] = topic_info['Topic'].apply(
            lambda x: self.cluster_descriptions.get(x, {}).get('name', '')
        )
        topic_info['description'] = topic_info['Topic'].apply(
            lambda x: self.cluster_descriptions.get(x, {}).get('description', '')
        )
        
        return topic_info
    
    def save_model(self, path: str = "bertopic_model"):
        if self.topic_model:
            self.topic_model.save(path)
            with open(f"{path}_descriptions.json", "w", encoding="utf-8") as f:
                json.dump(self.cluster_descriptions, f, ensure_ascii=False, indent=2)
            logger.info(f"Модель и описания сохранены в {path}")
        else:
            logger.warning("Нет обученной модели для сохранения")
    
    def load_model(self, path: str = "bertopic_model"):
        try:
            self.topic_model = BERTopic.load(path)
            with open(f"{path}_descriptions.json", "r", encoding="utf-8") as f:
                self.cluster_descriptions = json.load(f)
            logger.info(f"Модель и описания загружены из {path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
    
    def predict_topics(self, new_texts: List[str]) -> np.ndarray:
        if self.topic_model is None:
            logger.error("Модель не обучена")
            return np.array([])
        
        processed_texts = self.preprocess_texts(new_texts)
        topics, _ = self.topic_model.transform(processed_texts)
        return topics
    
    def save_cluster_names(self, path: str = "cluster_names.json"):
        if not self.cluster_descriptions:
            logger.warning("Нет описаний кластеров для сохранения")
            return
        
        cluster_names = {str(topic_id): info.get('name', '') 
                        for topic_id, info in self.cluster_descriptions.items()}
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cluster_names, f, ensure_ascii=False, indent=2)
            logger.info(f"Названия кластеров сохранены в {path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении названий кластеров: {e}")