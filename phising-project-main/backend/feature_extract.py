import re
import tldextract
import numpy as np

def extract_features(url):
    """
    Lexical-style feature extractor that returns a 30-element numeric array.
    Keep order & count identical to what you used during training.
    """

    # Ensure a scheme for parsing
    if not url.lower().startswith(("http://", "https://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    ext = tldextract.extract(url_for_parse)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    registered = ext.registered_domain or ""

    # Basic counts
    url_length = len(url)
    hostname_length = len(registered)
    # path length after the domain
    try:
        path = url_for_parse.split("/", 3)[-1]
        path_length = len(path) if "/" in url_for_parse else 0
    except:
        path_length = 0

    dot_count = url.count('.')
    slash_count = url.count('/')
    dash_count = url.count('-')
    special_chars = len(re.findall(r'[!@#$%^&*()?:;,_+%\\\[\]]', url))
    digit_count = len(re.findall(r'\d', url))
    letter_count = len(re.findall(r'[A-Za-z]', url))

    # IP address usage
    ip_address = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0

    # HTTPS flag
    https = 1 if url.lower().startswith("https://") else 0

    # Suspicious keywords
    suspicious_keywords = ["login", "secure", "update", "bank", "billing", "verify", "account", "signin", "confirm"]
    keyword_flag = 1 if any(k in url.lower() for k in suspicious_keywords) else 0

    # Subdomain count (levels)
    subdomain_count = 0 if subdomain == "" else len(subdomain.split("."))

    # Assemble features (lexical subset). Order matters.
    features = [
        url_length, hostname_length, path_length, dot_count, slash_count,
        dash_count, special_chars, digit_count, letter_count, ip_address,
        https, keyword_flag, subdomain_count
    ]

    # Pad with zeros to reach 30 features (keep order stable)
    while len(features) < 30:
        features.append(0.0)

    return np.array(features, dtype=float)
