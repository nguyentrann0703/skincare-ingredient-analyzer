"""
Paula's Choice - Full Detail Scraper
Scrape ALL ingredients with complete metadata
"""

from scrapling.spiders import Spider, Response
from scrapling.fetchers import AsyncDynamicSession
import json
import re

class PaulaChoiceFullSpider(Spider):
    name = "paula_choice_full"
    start_urls = ["https://www.paulaschoice.com/ingredient-dictionary"]
    concurrent_requests = 2
    
    def configure_sessions(self, manager):
        manager.add(
            "dynamic",
            AsyncDynamicSession(
                headless=True,
                network_idle=True,
                timeout=60000,
            ),
            default=True
        )
    
    @staticmethod
    def extract_ingredient_name(url):
        """Extract ingredient name from URL slug"""
        match = re.search(r'ingredient-dictionary/(.+?)(?:\.|$)', url)
        if not match:
            return None
        slug = match.group(1).replace('ingredient-', '').split('.')[0].split('?')[0]
        name = slug.replace('-', ' ')
        return ' '.join(word.capitalize() for word in name.split())
    
    async def parse(self, response: Response):
        """Parse list page - extract ingredient links and follow to detail pages"""
        for link in response.css('a[href*="/ingredient-dictionary/"]'):
            href = link.attrib.get('href', '')
            if href and '.html' in href:
                yield response.follow(href, callback=self.parse_detail)
        
        # Follow pagination
        next_page = (
            response.css('a[rel="next"]::attr(href)').get() or
            response.css('a.next::attr(href)').get() or
            response.css('[class*="next"] a::attr(href)').get()
        )
        
        if next_page:
            print(f"📄 Following next page...")
            yield response.follow(next_page)
    
    async def parse_detail(self, response: Response):
        """Parse detail page - extract ALL metadata"""
        ingredient_name = self.extract_ingredient_name(response.url)
        
        data = {
            'name': ingredient_name,
            'url': response.url,
            'title': response.css('title::text').get(),
        }
        
        # H1 Title
        h1 = response.css('h1::text').get()
        if h1:
            data['h1_title'] = h1.strip()
        
        # Rating
        rating_text = response.css('[class*="rating"]::text').get()
        if rating_text:
            data['rating'] = rating_text.strip()
        else:
            rating_class = response.css('[class*="rating"]::attr(class)').get()
            if rating_class:
                if 'good' in rating_class.lower():
                    data['rating'] = 'Good'
                elif 'average' in rating_class.lower():
                    data['rating'] = 'Average'
                elif 'poor' in rating_class.lower():
                    data['rating'] = 'Poor'
        
        # Main Description
        description = response.css('p::text').get()
        if description:
            data['description'] = description.strip()
        
        # Full Description
        all_descriptions = response.css('p::text').getall()
        if all_descriptions:
            data['full_description'] = ' '.join([p.strip() for p in all_descriptions])
        
        # Benefits
        benefits = response.css('[class*="benefit"], [class*="function"]::text').getall()
        if benefits:
            data['benefits'] = [b.strip() for b in benefits if b.strip()]
        
        # Usage
        usage_text = response.css('[class*="usage"], [class*="concentration"]::text').getall()
        if usage_text:
            data['usage'] = [u.strip() for u in usage_text if u.strip()]
        
        # Safety
        safety_text = response.css('[class*="safe"], [class*="safety"]::text').getall()
        if safety_text:
            data['safety_notes'] = [s.strip() for s in safety_text if s.strip()]
        
        # Other Names
        other_names = response.css('[class*="other"], [class*="also"]::text').getall()
        if other_names:
            data['other_names'] = [n.strip() for n in other_names if n.strip()]
        
        # Related Ingredients
        related_links = response.css('a[href*="/ingredient-dictionary/ingredient-"]::attr(href)').getall()
        if related_links:
            related_names = [self.extract_ingredient_name(url) for url in related_links]
            data['related_ingredients'] = [n for n in related_names if n]
        
        # Meta Description
        meta_desc = response.xpath('//meta[@name="description"]/@content').get()
        if meta_desc:
            data['meta_description'] = meta_desc
        
        # OG Description
        og_desc = response.xpath('//meta[@property="og:description"]/@content').get()
        if og_desc:
            data['og_description'] = og_desc
        
        yield data


if __name__ == "__main__":
    print("🚀 Starting Paula's Choice FULL DETAIL Scraper...")
    print("⏳ This will take a while - scraping ALL ingredients with full metadata...\n")
    
    result = PaulaChoiceFullSpider().start()
    
    result.items.to_json("paula_choice_full_details.json")
    
    print("\n" + "="*80)
    print(f"✅ Scraped {len(result.items)} ingredient DETAILS")
    print(f"💾 Saved to: paula_choice_full_details.json")
    # ✅ FIX: Use dot notation instead of dictionary access
    print(f"📊 Elapsed time: {result.stats.elapsed_seconds:.2f}s")
    print("="*80)
    
    if result.items:
        with open("paula_choice_full_details.json", "r", encoding='utf-8') as f:
            data = json.load(f)
            
            print(f"\n📋 Sample 1 - Full Detail View:")
            print("-" * 80)
            sample = data[0]
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"{key}: {value[:2] if len(value) > 2 else value}")
                elif isinstance(value, str) and len(str(value)) > 100:
                    print(f"{key}: {str(value)[:100]}...")
                else:
                    print(f"{key}: {value}")
            
            print(f"\n\n📊 Statistics:")
            print("-" * 80)
            print(f"Total ingredients: {len(data)}")
            print(f"Ingredients with rating: {sum(1 for d in data if 'rating' in d)}")
            print(f"Ingredients with description: {sum(1 for d in data if 'description' in d)}")
            print(f"Ingredients with benefits: {sum(1 for d in data if 'benefits' in d)}")
            print(f"Ingredients with related: {sum(1 for d in data if 'related_ingredients' in d)}")