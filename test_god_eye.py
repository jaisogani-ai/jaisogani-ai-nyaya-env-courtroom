import requests
import time

BASE_URL = "http://localhost:7860"

def test_inject():
    # 1. Reset env
    print("Resetting environment...")
    resp = requests.post(f"{BASE_URL}/reset", json={"task": "medium"})
    obs = resp.json()
    print(f"Initial Evidence: {obs.get('evidence_strength')}")
    
    # 2. Test injection
    event = "New evidence found"
    print(f"\nInjecting event: '{event}'")
    resp = requests.post(f"{BASE_URL}/inject", json={"event": event})
    data = resp.json()
    
    if "error" in data:
        print(f"Error: {data['error']}")
        return

    print(f"Impact: {data['impact']}")
    print(f"Prob Before: {data['before']}%")
    print(f"Prob After: {data['after']}%")
    print(f"Change: {data['change']}%")
    print(f"New Evidence Strength: {data['observation'].get('evidence_strength')}")

    # 3. Test Art 21 injection
    event = "Article 21 emergency invoked"
    print(f"\nInjecting event: '{event}'")
    resp = requests.post(f"{BASE_URL}/inject", json={"event": event})
    data = resp.json()
    print(f"Impact: {data['impact']}")
    print(f"Prob Before: {data['before']}%")
    print(f"Prob After: {data['after']}%")
    print(f"Change: {data['change']}%")
    print(f"Article 21 Breached: {data['observation'].get('article21_threshold_breached')}")

    # 4. Test PMLA Ruling
    event = "New Supreme Court PMLA ruling"
    print(f"\nInjecting event: '{event}'")
    resp = requests.post(f"{BASE_URL}/inject", json={"event": event})
    data = resp.json()
    print(f"Impact: {data['impact']}")
    print(f"Prob Before: {data['before']}%")
    print(f"Prob After: {data['after']}%")
    print(f"Change: {data['change']}%")
    print(f"Defense Score: {data['observation'].get('defense_score')}")

    # 5. Test Judge Conflict
    event = "Judge conflict of interest"
    print(f"\nInjecting event: '{event}'")
    resp = requests.post(f"{BASE_URL}/inject", json={"event": event})
    data = resp.json()
    print(f"Impact: {data['impact']}")
    # This one doesn't affect probability directly in the user's formula, but let's check it doesn't crash
    print(f"Clerk Warnings: {data['observation'].get('clerk_warnings')}")
    print(f"Oversight Budget: {data['observation'].get('oversight_budget')}")

if __name__ == "__main__":
    try:
        test_inject()
    except Exception as e:
        print(f"Test failed: {e}")
