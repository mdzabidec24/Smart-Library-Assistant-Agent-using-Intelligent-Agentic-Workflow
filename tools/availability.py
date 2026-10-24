from .book_search import search_books
from .recommend import recommend_books
import pandas as pd

def check_availability(query: str, df=None) -> str:
    if df is None:
        try:
            df = pd.read_csv("data/library.csv")
        except:
            return "No data"
    res = search_books(query, df=df)
    if isinstance(res, list) and len(res)==1:
        c = int(res[0]['available_copies'])
        rec = recommend_books(query, df, current_title=res[0]['title'])
        if c==0:
            return f"0 copies - will be available soon at Rack {res[0]['rack_no']}{rec}"
        return f"{c} copies at Rack {res[0]['rack_no']}{rec}"
    return str(res)