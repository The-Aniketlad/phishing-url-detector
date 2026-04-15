# backend/suggestions.py
from safe_domains import SAFE_DOMAINS

def suggest_alternative(url):
    url_lower = url.lower()
    suggestions = []

    for brand, safe_url in SAFE_DOMAINS.items():
        if brand in url_lower:
            suggestions.append(safe_url)

    return suggestions
