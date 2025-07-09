
import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    # Test de santé
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API Health Check: OK")
        else:
            print("❌ API Health Check: FAILED")
    except:
        print("❌ API non accessible")
        return False
    
    # Test de recommandation
    try:
        response = requests.post(f"{base_url}/recommendations", 
                               json={"user_id": 1, "n_recommendations": 5})
        if response.status_code == 200:
            print("✅ API Recommendations: OK")
            recommendations = response.json()
            print(f"   Recommandations reçues: {len(recommendations.get('recommendations', []))}")
        else:
            print("❌ API Recommendations: FAILED")
    except Exception as e:
        print(f"❌ Erreur test recommandations: {e}")
    
    return True

if __name__ == "__main__":
    test_api()
