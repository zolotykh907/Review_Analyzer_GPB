import pandas as pd
import json
import html
import re
from bs4 import BeautifulSoup

def clean_html_text(html_text):
    decoded = html.unescape(html_text)
    soup = BeautifulSoup(decoded, "html.parser")
    return soup.get_text(" ", strip=True)

def fix_json_escape_sequences(json_string):
    fixed_json = re.sub(r'\\(?!["/\\bfnrtu])', r'\\\\', json_string)
    return fixed_json

def clean_json_string(json_string):
    json_string = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_string)
    json_string = fix_json_escape_sequences(json_string)
    json_string = re.sub(r'\s+', ' ', json_string).strip()
    
    return json_string

def parse_reviews(html_text):
    soup = BeautifulSoup(html_text, "html.parser")

    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        raise ValueError("JSON-LD not found")

    raw_json = script_tag.string.strip()
    
    cleaned_json = clean_json_string(raw_json)
    
    try:
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
        print(f"Проблемный фрагмент: {cleaned_json[e.pos-50:e.pos+50]}")
        return []
    
    reviews = data.get("review", [])

    parsed = []
    for r in reviews:
        try:
            bank = r.get("itemReviewed", {}).get("name", "")
            if bank == 'Газпромбанк':
                title = r.get("name", "").strip()
                body_html = r.get("reviewBody", "")
                body_text = clean_html_text(body_html)
                rating = r.get("reviewRating", {}).get("ratingValue")
                date = r.get("datePublished")

                parsed.append({
                    "title": title,
                    "text": body_text,
                    "rating": rating,
                    "date": date
                })
        except (KeyError, AttributeError) as e:
            print(f"Ошибка обработки отзыва: {e}")
            continue

    return parsed

def safe_request(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp
            else:
                print(f"Страница {url}: статус {resp.status_code}")
        except requests.RequestException as e:
            print(f"Ошибка запроса {url} (попытка {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return None

if __name__ == "__main__":
    import requests
    import time

    all_parsed_reviews = []
    pages_count = 500

    for i in range(1, pages_count + 1):
        url = f"https://www.banki.ru/services/responses/list/?page={i}&is_countable=on"
        print(f"Обрабатывается страница {i}/{pages_count}...")
        
        resp = safe_request(url)
        
        if resp:
            try:
                parsed_reviews = parse_reviews(resp.text)
                all_parsed_reviews.extend(parsed_reviews)
                print(f"Найдено отзывов на странице: {len(parsed_reviews)}")
            except Exception as e:
                print(f"Ошибка парсинга страницы {i}: {e}")
                continue
        else:
            print(f"Не удалось загрузить страницу {i}")

    if all_parsed_reviews:
        df = pd.DataFrame(all_parsed_reviews)
        df.to_json("data/reviews_banki_ru.json", orient="records", force_ascii=False, indent=2)
        print(f"Сохранено {len(all_parsed_reviews)} отзывов")
        
        print(f"\nСтатистика:")
        print(f"Всего отзывов: {len(df)}")
    else:
        print("Не найдено отзывов для сохранения")