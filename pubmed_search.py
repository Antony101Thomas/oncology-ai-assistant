import requests
import xml.etree.ElementTree as ET

def search_pubmed(query: str, max_results: int = 3) -> list:
    print(f"Searching PubMed for: {query}")
    
    # Step 1: Search for article IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    
    try:
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        ids = search_response.json()["esearchresult"]["idlist"]
    except requests.RequestException as exc:
        print(f"PubMed search failed: {exc}")
        return []
    
    if not ids:
        print("No results found!")
        return []
    
    print(f"Found {len(ids)} articles!")
    
    # Step 2: Fetch article details
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }
    
    try:
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_response.raise_for_status()
    except requests.RequestException as exc:
        print(f"PubMed fetch failed: {exc}")
        return []

    root = ET.fromstring(fetch_response.content)
    
    articles = []
    for article in root.findall(".//PubmedArticle"):
        # Get title
        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else "No title"
        
        # Get abstract
        abstract_el = article.find(".//AbstractText")
        abstract = abstract_el.text if abstract_el is not None else "No abstract"
        
        # Get PubMed ID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        
        articles.append({
            "title": title,
            "abstract": abstract[:500],  # first 500 chars
            "source": f"PubMed ID: {pmid}",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })
    
    return articles
