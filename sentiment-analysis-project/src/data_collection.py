import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AmazonReviewScraper:
    def __init__(self):
        # Using a list of user agents to avoid getting blocked
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]

    def get_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept-Language': 'en-US, en;q=0.5'
        }

    def get_reviews_from_page(self, url):
        """Scrape reviews from a single Amazon product reviews page."""
        try:
            response = requests.get(url, headers=self.get_headers())
            if response.status_code != 200:
                logging.warning(f"Failed to fetch page. Status code: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            reviews = soup.find_all('div', {'data-hook': 'review'})
            
            parsed_reviews = []
            for review in reviews:
                try:
                    rating = review.find('i', {'data-hook': 'review-star-rating'}).text.split('.')[0]
                    title = review.find('a', {'data-hook': 'review-title'}).text.strip()
                    body = review.find('span', {'data-hook': 'review-body'}).text.strip()
                    
                    parsed_reviews.append({
                        'rating': int(rating),
                        'title': title,
                        'text': body
                    })
                except Exception as e:
                    logging.debug(f"Error parsing a review: {e}")
                    continue
                    
            return parsed_reviews
            
        except Exception as e:
            logging.error(f"Error requesting page: {e}")
            return []

    def scrape_product(self, base_url, max_pages=5):
        """Scrape multiple pages of reviews for a product."""
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            url = f"{base_url}&pageNumber={page}"
            logging.info(f"Scraping page {page}...")
            
            reviews = self.get_reviews_from_page(url)
            if not reviews:
                break
                
            all_reviews.extend(reviews)
            
            # Be polite and sleep between requests
            time.sleep(random.uniform(2.0, 5.0))
            
        return pd.DataFrame(all_reviews)

if __name__ == "__main__":
    # Example usage
    scraper = AmazonReviewScraper()
    # Replace with a real Amazon review page URL
    # sample_url = "https://www.amazon.com/product-reviews/B08N5WRWNW/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
    # df = scraper.scrape_product(sample_url, max_pages=2)
    # print(df.head())
    # df.to_csv('../data/scraped_reviews.csv', index=False)
    print("Scraper ready! Remember that Amazon frequently blocks simple requests. For large scale scraping, consider using APIs or Selenium.")
