import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "backend"))
from feature_extract import extract_features

print("==================================================")
print("Running URL Feature Extractor Unit Tests...")
print("==================================================")

# Test inputs
test_cases = [
    "http://www.google.com",
    "https://secure-paypal-login-update.com/signin",
    "http://192.168.1.1/login"
]

for i, url in enumerate(test_cases, 1):
    try:
        features = extract_features(url)
        # Verify shape (our model expects exactly 30 padded features)
        assert len(features) == 30, f"Feature dimension mismatch: expected 30, got {len(features)}"
        
        # Verify specific feature checks
        assert features[0] == len(url), "URL length extraction failed"
        
        print(f"Test Case {i}: {url}")
        print(f" -> Features (first 5 values): {features[:5]}... [PASS]")
    except AssertionError as ae:
        print(f"Test Case {i}: {url} -> [FAIL] ({str(ae)})")
        sys.exit(1)
    except Exception as e:
        print(f"Test Case {i}: {url} -> [ERROR] ({str(e)})")
        sys.exit(1)

print("--------------------------------------------------")
print("STATUS: ALL FEATURE EXTRACTION TESTS PASSED [100%]")
print("==================================================")
