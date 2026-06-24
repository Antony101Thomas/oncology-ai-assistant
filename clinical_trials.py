import requests

def search_clinical_trials(query: str, max_results: int = 3) -> list:
    print(f"Searching ClinicalTrials.gov for: {query}")
    
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": query,
        "pageSize": max_results,
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"ClinicalTrials.gov search failed: {exc}")
        return []
    
    studies = data.get("studies", [])
    
    if not studies:
        print("No trials found!")
        return []
    
    print(f"Found {len(studies)} trials!")
    
    trials = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        desc_module = protocol.get("descriptionModule", {})
        
        nct_id = id_module.get("nctId", "")
        title = id_module.get("briefTitle", "No title")
        status = status_module.get("overallStatus", "Unknown")
        description = desc_module.get("briefSummary", "No description")
        
        trials.append({
            "title": title,
            "status": status,
            "description": description[:400],
            "source": f"ClinicalTrials.gov - {nct_id}",
            "url": f"https://clinicaltrials.gov/study/{nct_id}"
        })
    
    return trials
