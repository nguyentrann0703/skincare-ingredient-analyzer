"""
Paula's Choice - Full Detail Scraper
Uses ajax=true API endpoint — no browser needed, ~20x faster
"""

from scrapling.spiders import Spider, Request, Response
from scrapling.fetchers import FetcherSession
import json
import re

BASE_URL = "https://www.paulaschoice.com"
AJAX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "identity",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}


def extract_text(obj) -> str:
    """Đệ quy extract text từ bất kỳ cấu trúc nào: str, dict, list lồng nhau"""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("text", "")
    if isinstance(obj, list):
        return " ".join(extract_text(item) for item in obj if item)
    return ""


class PaulaChoiceFullSpider(Spider):
    name = "paula_choice_full"
    start_urls = [
        # ⚠️ TEST MODE: chỉ lấy 10 ingredients đầu — xóa [:10] khi scrape full
        "https://www.paulaschoice.com/ingredient-dictionary"
        "?crefn1=ingredientRating&crefv1=Best%7CGood%7CAverage%7CBad%7CWorst"
        "&csortb1=name&csortd1=1&start=0&sz=2000&ajax=true"
    ]
    concurrency = 5
    download_delay = 0.5

    def configure_sessions(self, manager):
        manager.add(
            "http",
            FetcherSession(impersonate="chrome"),
            default=True
        )

    async def parse(self, response: Response):
        try:
            raw = response.text if response.text.strip() else response.body.decode('utf-8')
            d = json.loads(raw)
        except Exception as e:
            print(f"❌ Failed to parse JSON from {response.url}: {e}")
            return

        ingredients = d.get("ingredients", [])
        print(f"📄 Found {len(ingredients)} ingredients on this page")

        for item in ingredients:
            ingredient_id = item.get("id", "")
            if ingredient_id:
                ajax_url = f"{BASE_URL}/ingredient-dictionary/{ingredient_id}.html?ajax=true"
                yield Request(ajax_url, callback=self.parse_detail, headers=AJAX_HEADERS)

        # Generate remaining pages từ trang đầu
        paging = d.get("paging", {})
        total = paging.get("total", 0)
        current_start = paging.get("start", 0)

        if current_start == 0 and total > 2000:
            base = (
                f"{BASE_URL}/ingredient-dictionary"
                "?crefn1=ingredientRating&crefv1=Best%7CGood%7CAverage%7CBad%7CWorst"
                "&csortb1=name&csortd1=1&sz=2000&ajax=true"
            )
            print(f"📊 Total rated ingredients: {total} — generating {total // 2000} more pages...")
            for start in range(2000, total, 2000):
                yield Request(f"{base}&start={start}", callback=self.parse, headers=AJAX_HEADERS)

    async def parse_detail(self, response: Response):
        try:
            raw = response.text if response.text.strip() else response.body.decode('utf-8')
            d = json.loads(raw)
        except Exception:
            print(f"❌ Failed to parse JSON from {response.url}")
            return

        # Description — dùng extract_text() để handle mọi cấu trúc lồng nhau
        description_parts = []
        for para in d.get("description", []):
            for text_item in para.get("text", []):
                extracted = extract_text(text_item).strip()
                if extracted:
                    description_parts.append(extracted)
        description = " ".join(description_parts)

        data = {
            "h1_title": d.get("name"),
            "url": f"{BASE_URL}/ingredient-dictionary/{d.get('id', '')}.html",
            "rating": d.get("rating"),
            "rating_value": d.get("ratingValue"),
            "description": description,
            "key_points": d.get("keyPoints", []),
            "benefits": [b.get("name") for b in d.get("benefits", []) if b.get("name")],
            "categories": [c.get("name") for c in d.get("relatedCategories", []) if c.get("name")],
            "related_ingredients": [r.get("name") for r in d.get("related", []) if r.get("name")],
            "references": d.get("references", []),
            "authors": [a.get("name") for a in d.get("authors", []) if a.get("name")],
            "reviewers": [r.get("name") for r in d.get("reviewers", []) if r.get("name")],
            "date_modified": d.get("dateModified"),
        }

        yield data


if __name__ == "__main__":
    print("🚀 Starting Paula's Choice FULL DETAIL Scraper (API mode)...")
    print("⚡ No browser needed — using ajax=true API\n")

    result = PaulaChoiceFullSpider().start()
    result.items.to_json("paula_choice_full_details.json")

    print("\n" + "=" * 80)
    print(f"✅ Scraped {len(result.items)} ingredient DETAILS")
    print(f"💾 Saved to: paula_choice_full_details.json")
    print(f"📊 Elapsed time: {result.stats.elapsed_seconds:.2f}s")
    print("=" * 80)

    if result.items:
        with open("paula_choice_full_details.json", "r", encoding="utf-8") as f:
            data = json.load(f)

            print(f"\n📋 Sample - Full Detail View:")
            print("-" * 80)
            sample = data[0]
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"{key}: {value[:3] if len(value) > 3 else value}")
                elif isinstance(value, str) and len(value) > 120:
                    print(f"{key}: {value[:120]}...")
                else:
                    print(f"{key}: {value}")

            print(f"\n\n📊 Statistics:")
            print("-" * 80)
            print(f"Total ingredients:            {len(data)}")
            print(f"Ingredients with rating:      {sum(1 for d in data if d.get('rating'))}")
            print(f"Ingredients with description: {sum(1 for d in data if d.get('description'))}")
            print(f"Ingredients with key_points:  {sum(1 for d in data if d.get('key_points'))}")
            print(f"Ingredients with benefits:    {sum(1 for d in data if d.get('benefits'))}")
            print(f"Ingredients with related:     {sum(1 for d in data if d.get('related_ingredients'))}")
            print(f"Ingredients with references:  {sum(1 for d in data if d.get('references'))}")