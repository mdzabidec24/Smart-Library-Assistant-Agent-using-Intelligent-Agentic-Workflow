import pandas as pd
import re
from langchain_core.tools import tool

def search_books_core(query, df, llm=None):
    """Core search for books - internal use"""
    q = query.lower().strip()
    q_norm = re.sub(r'\bbooks\b', 'book', q)
    q_norm = re.sub(r'\bmicroprocessor\b', 'microprocessors', q_norm)
    
    stop = {"what","are","th","the","books","book","related","to","is","a","an","how","many","there","of","amny","copies","it","available","i","there","are","there","genre","where","is","the","we","have","and","of","who","author","that","this","where"}

    # TITLE FIRST - Fixes overlap
    for _, row in df.iterrows():
        title = str(row['title']).lower().strip()
        if title in q or title in q_norm:
            if len(title) > 3:
                return [row.to_dict()]
    
    q_words = set(re.findall(r'\w+', q_norm)) - stop
    for _, row in df.iterrows():
        title = str(row['title']).lower()
        title_words = set(re.findall(r'\w+', title))
        if title_words and title_words.issubset(q_words) and len(title_words) >= 2:
            return [row.to_dict()]

    # CATEGORY ONLY AFTER TITLE
    if "computer science" in q or "computer scence" in q:
        if "networks" not in q and "programming" not in q or "books related to" in q:
            if "computer networks" not in q and "python programming" not in q and "ios programming" not in q:
                filtered = df[df['category'].str.lower() == "computer science"]
                if not filtered.empty:
                    return filtered.to_dict('records')

    categories = df['category'].astype(str).str.lower().unique()
    for cat in categories:
        if cat != "computer science" and cat in q and len(cat) > 2:
            filtered = df[df['category'].str.lower() == cat]
            if not filtered.empty:
                return filtered.to_dict('records')

    # PARTIAL - Need 2 words to avoid overlap
    if len(q_words) >= 2:
        matches = []
        for _, row in df.iterrows():
            title_words = set(re.findall(r'\w+', str(row['title']).lower()))
            intersect = q_words & title_words
            if intersect and len(intersect) >= 2:
                matches.append((len(intersect), row.to_dict()))
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            return [matches[0][1]]

    return f"No books found: {query}"

@tool
def library_search_tool(query: str) -> str:
    """Search for books in library by title, author or category."""
    try:
        df = pd.read_csv("data/library.csv")
        result = search_books_core(query, df)
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            out = []
            for r in result:
                out.append(f"{r['title']} by {r['author']} - {r['available_copies']} copies at Rack {r['rack_no']}")
            return "; ".join(out)
        return "No books found"
    except Exception as e:
        return f"Error: {e}"