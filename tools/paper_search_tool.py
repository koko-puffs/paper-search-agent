from semanticscholar import SemanticScholar
from typing import List, Dict, Any, Optional

def search_papers(
    topic: str,
    year: Optional[int] = None,
    year_operator: Optional[str] = None,
    min_citations: Optional[int] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    sch = SemanticScholar()
    query = topic
    
    year_filter_str = None
    if year is not None:
        if year_operator == "in":
            year_filter_str = str(year)
        elif year_operator == "before":
            year_filter_str = f"-{year-1}" 
        elif year_operator == "after":
            year_filter_str = f"{year+1}-"
        else:
            year_filter_str = str(year)
    
    results = sch.search_paper(
        query=query,
        year=year_filter_str,
        limit=limit,
        min_citation_count=min_citations,
    )

    papers = []
    if results:
        for item in results:
            if len(papers) >= limit:
                break

            paper_data = {
                "title": getattr(item, 'title', "N/A"),
                "year": getattr(item, 'year', None),
                "authors": [getattr(author, 'name', "N/A") for author in getattr(item, 'authors', [])],
                "abstract": getattr(item, 'abstract', "N/A"),
                "url": getattr(item, 'url', "N/A"),
                "citationCount": getattr(item, 'citationCount', 0),
                "paperId": getattr(item, 'paperId', "N/A"),
            }
            papers.append(paper_data)
            
    return papers 