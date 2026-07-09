import requests, re
from bs4 import BeautifulSoup

url = "https://www.flipkart.com/realme-p4x-5g-matte-silver-128-gb/product-reviews/itm575b1540859e4?pid=MOBHHWGMCXRADYYG"

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/126.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}

print("Fetching with requests...")
try:
    r = requests.get(url, headers=HEADERS, timeout=15)
    print(f"Status code: {r.status_code}")
    print(f"HTML length: {len(r.text)}")
    
    soup = BeautifulSoup(r.text, 'lxml')
    # Search for reviews using our new css- pattern
    count = 0
    seen = set()
    for tag in soup.find_all(['div', 'span', 'p']):
        classes = tag.get('class', [])
        cls_str = ' '.join(classes)
        if any(c.startswith('css-') for c in classes) or 'css-' in cls_str:
            text = tag.get_text(strip=True)
            if 15 < len(text) < 1000 and text not in seen:
                text_lower = text.lower()
                if any(kw in text_lower for kw in ['outer ring road', 'flipkart internet', 'help center', 'shipping', 'policy']):
                    continue
                if len(tag.find_all('div')) > 2:
                    continue
                seen.add(text)
                print(f"  [{tag.name}.{cls_str[:15]}] {text[:100]}")
                count += 1
    print(f"Found {count} reviews.")
except Exception as e:
    print(f"Error: {e}")
