# llm_client.py
import requests
import re
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class LMStudioClient:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions", 
                 model: str = "qwen3-4b-2507",
                 temperature: float = 0.3,
                 max_tokens: int = 4096):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _call_api(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    logger.warning(f"Ошибка LLM API (попытка {attempt + 1}): {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ошибка подключения к LLM (попытка {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
        
        logger.error("Не удалось получить ответ от LLM после всех попыток")
        return ""
    
    def _create_topic_prompt(self, keywords: list, sample_texts: list) -> str:
        return f"""
Ты — эксперт по анализу банковских отзывов. На основе ключевых слов и примеров отзывов создай максимально обобщённое и короткое название темы (одно слово, максимум три-четыре) и её описание.

Ключевые слова темы: {', '.join(keywords[:10])}

Примеры отзывов из этой темы:
{chr(10).join(['- ' + text[:200] + '...' for text in sample_texts[:3]])}

Верни ответ ТОЛЬКО в формате JSON:
{{
    "name": "Обобщённое название темы (1-4 слова)",
    "description": "Подробное описание темы (1-2 предложения)"
}}

Название должно быть максимально коротким, обобщённым и отражать суть темы. Используй только банковскую терминологию.
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Парсинг ответа от LLM"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "name": result.get("name", "").strip(),
                    "description": result.get("description", "").strip()
                }
        except Exception as e:
            logger.warning(f"Не удалось распарсить JSON от LLM: {e}")
        
        return {}
    
    def generate_topic_name_and_description(self, keywords: list, sample_texts: list) -> Dict[str, str]:
        """Генерация названия и описания темы"""
        if not keywords or not sample_texts:
            return {}
        
        prompt = self._create_topic_prompt(keywords, sample_texts)
        response = self._call_api(prompt)
        
        if not response:
            return {}
        
        return self._parse_llm_response(response)