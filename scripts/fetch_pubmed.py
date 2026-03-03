from Bio import Entrez
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

DISEASE_QUERIES = {
    "diabetes": "type 2 diabetes[Title/Abstract]",
    "depression": "depression treatment[Title/Abstract]",
    "cardiovascular": "cardiovascular disease[Title/Abstract]"
}

def fetch_pmids(query, max_results=1500):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_abstracts(pmids, disease_area):
    papers = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records["PubmedArticle"]:
            try:
                medline = article["MedlineCitation"]
                art = medline["Article"]

                if "Abstract" not in art:
                    continue
                abstract_texts = art["Abstract"]["AbstractText"]
                if isinstance(abstract_texts, list):
                    abstract = " ".join([str(t) for t in abstract_texts])
                else:
                    abstract = str(abstract_texts)

                if len(abstract) < 100:
                    continue

                authors = []
                if "AuthorList" in art:
                    for author in art["AuthorList"]:
                        if "LastName" in author:
                            authors.append(author["LastName"])

                year = None
                if "Journal" in art:
                    pub_date = art["Journal"]["JournalIssue"]["PubDate"]
                    year = str(pub_date.get("Year", pub_date.get("MedlineDate", "Unknown")))[:4]

                paper = {
                    "pmid": str(medline["PMID"]),
                    "title": str(art["ArticleTitle"]),
                    "abstract": abstract,
                    "authors": authors[:5],
                    "year": year,
                    "journal": str(art["Journal"]["Title"]),
                    "disease_area": disease_area
                }
                papers.append(paper)

            except Exception:
                continue

        print(f"  Batch {i//100 + 1} done, total so far: {len(papers)}")
        time.sleep(0.34)

    return papers

def main():
    all_papers = []

    for disease, query in DISEASE_QUERIES.items():
        print(f"\nFetching: {disease}")
        pmids = fetch_pmids(query, max_results=1500)
        print(f"  Found {len(pmids)} PMIDs")
        papers = fetch_abstracts(pmids, disease_area=disease)
        print(f"  Got {len(papers)} papers with abstracts")
        all_papers.extend(papers)

    with open("data/papers.json", "w") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\nDone. Total saved: {len(all_papers)}")
    print("File: data/papers.json")

if __name__ == "__main__":
    main()