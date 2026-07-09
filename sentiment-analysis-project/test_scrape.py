"""Quick debug: fetch Flipkart reviews page with Selenium and inspect HTML."""
import time, re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

url = "https://www.flipkart.com/realme-p4x-5g-matte-silver-128-gb/product-reviews/itm575b1540859e4?pid=MOBHHWGMCXRADYYG"

print("Reading saved HTML...")
with open('data/flipkart_debug.html', 'r', encoding='utf-8') as f:
    page_source = f.read()
soup = BeautifulSoup(page_source, 'lxml')
print(f"HTML length: {len(page_source)}")

import re

def is_valid_review(text):
    if len(text) < 15 or len(text) > 2000:
        return False
        
    unwanted_keywords = [
        'flipkart internet private limited',
        'flipkart private limited',
        'flipkart internet',
        'amazon.com',
        'seller information',
        'shipping address',
        'phone number',
        'email address',
        'terms & conditions',
        'privacy policy',
        'return policy',
        'warrant',
        'disclaimer',
        'call us',
        'contact us',
        'your cart',
        'out of stock',
        'verified purchase',
        'verified buyer',
        'helpful for',
        'report abuse',
        'months ago',
        'days ago',
        'years ago',
        'customer reviews sorted by',
        'outer ring road',
        'devarabeesanahalli village',
        'flipkart customer',
    ]
    
    text_lower = text.lower()
    for keyword in unwanted_keywords:
        if keyword in text_lower:
            return False
            
    if text.strip().startswith(','):
        return False
        
    if ',' in text:
        parts = text.split(',')
        if len(parts) == 2:
            loc_part = parts[1].strip()
            # If the location part is 1-2 words and starts with capital letter, it's a location
            if loc_part and len(loc_part.split()) <= 2 and loc_part[0].isupper():
                return False
            
    words = re.findall(r'\b[a-z]+\b', text_lower)
    if len(words) < 3:
        return False
        
    return True

# Run extraction logic
reviews = []
seen = set()

def is_valid_review(text):
    if len(text) < 15 or len(text) > 2000:
        return False
        
    unwanted_keywords = [
        'flipkart internet private limited',
        'flipkart private limited',
        'flipkart internet',
        'amazon.com',
        'seller information',
        'shipping address',
        'phone number',
        'email address',
        'terms & conditions',
        'privacy policy',
        'return policy',
        'warrant',
        'disclaimer',
        'call us',
        'contact us',
        'your cart',
        'out of stock',
        'verified purchase',
        'verified buyer',
        'helpful for',
        'report abuse',
        'months ago',
        'days ago',
        'years ago',
        'customer reviews sorted by',
        'outer ring road',
        'devarabeesanahalli village',
        'flipkart customer',
        'hang on, loading content',
        'loading content',
    ]
    
    text_lower = text.lower()
    for keyword in unwanted_keywords:
        if keyword in text_lower:
            return False
            
    if text.strip().startswith(','):
        return False
        
    # Filter out location strings
    if ',' in text:
        parts = text.split(',')
        if len(parts) == 2:
            loc_part = parts[1].strip()
            if loc_part and len(loc_part.split()) <= 2 and loc_part[0].isupper():
                return False
                
    # Filter out all-uppercase short names
    if len(text) < 30 and text.isupper():
        return False
            
    words = re.findall(r'\b[a-z]+\b', text_lower)
    if len(words) < 3:
        return False
        
    return True

def _add(text):
    text = text.strip()
    if text and text not in seen and is_valid_review(text):
        reviews.append(text)
        seen.add(text)

# 1. Try our new css-* Emotion selector
for tag in soup.find_all(['div', 'span', 'p']):
    classes = tag.get('class', [])
    cls_str = ' '.join(classes)
    if any(c.startswith('css-') for c in classes) or 'css-' in cls_str:
        text = tag.get_text(strip=True)
        if text.startswith('Review for:') or 'ratings and' in text.lower() or 'reviews sorted by' in text.lower():
            continue
        if len(tag.find_all('div')) > 2:
            continue
        _add(text)

print(f"Extracted {len(reviews)} reviews:")
for idx, r in enumerate(reviews):
    safe_r = r.encode('ascii', 'replace').decode('ascii')
    print(f"  {idx+1}: {safe_r}")

print("\nDone.")
