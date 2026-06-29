import sys
import os
sys.path.append(os.path.abspath("backend"))
from backend.app import app

with app.test_client() as client:
    resp = client.post("/predict", json={"url": "http://example.com"})
    print("URL SCAN:", resp.status_code, resp.json)

    resp2 = client.post("/predict-message", json={"message": "Win a free iphone!"})
    print("MSG SCAN:", resp2.status_code, resp2.json)
