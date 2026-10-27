import pandas as pd
from collections import Counter
import streamlit as st

def recommend_books(query: str, df=None, current_title=None):
    if 'book_count' not in st.session_state:
        st.session_state.book_count = Counter()
    
    track_title = current_title if current_title else query
    if not track_title:
        return ""
    
    key = str(track_title).lower().strip()
    if key in ["it","this","that","how much","how many","available now","is it available now","how much copies","how many copies"]:
        return ""
    
    st.session_state.book_count[key] += 1
    
    if st.session_state.book_count[key] >= 3:
        try:
            if df is None:
                df = pd.read_csv("data/library.csv")
            df['title'] = df['title'].astype(str).str.strip()
            df['category'] = df['category'].astype(str).str.strip()
            df['available_copies'] = pd.to_numeric(df['available_copies'], errors='coerce').fillna(0).astype(int)
            
            row = df[df['title'].str.lower() == key]
            if not row.empty:
                cat = row.iloc[0]['category']
                # BEST 2 = highest copies in same category, excluding current
                same_cat = df[(df['category'] == cat) & (df['title'].str.lower() != key)]
                same_cat = same_cat.sort_values(by='available_copies', ascending=False)
                if not same_cat.empty:
                    top2 = same_cat.head(2)['title'].tolist()
                    return f"\n\n📚 You asked about **{cat}** 3 times! You may also like: {', '.join(top2)}"
        except:
            pass
    return ""