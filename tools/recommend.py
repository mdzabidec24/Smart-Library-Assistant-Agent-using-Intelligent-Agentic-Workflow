import pandas as pd
import streamlit as st
from collections import defaultdict

def recommend_books(query: str, df=None, current_title=None):
    if df is None:
        try:
            df = pd.read_csv("data/library.csv")
        except:
            return ""
    if current_title and len(current_title) < 3:
        current_title = None
    q = (current_title or query).lower().strip()
    if len(q) < 3:
        return ""
    found = df[df['title'].str.lower()==q]
    if found.empty:
        found = df[df['title'].str.lower().str.contains(q, na=False, regex=False)]
    if found.empty:
        return ""
    
    # --- COUNT LOGIC FOR 3 TIMES ---
    if "topic_counts" not in st.session_state:
        st.session_state.topic_counts = defaultdict(int)
    
    cat = found.iloc[0]['category']
    title = found.iloc[0]['title']
    
    # Use category as topic
    topic_key = cat.lower()
    st.session_state.topic_counts[topic_key] += 1
    
    # Only recommend if asked 3 times about same category
    if st.session_state.topic_counts[topic_key] < 3:
        return ""
    
    # Reset after recommending so it doesn't spam
    st.session_state.topic_counts[topic_key] = 0
    
    rec = df[(df['category']==cat) & (df['title']!=title) & (df['available_copies']>0)]
    if rec.empty:
        rec = df[(df['category']==cat) & (df['title']!=title)]
    if not rec.empty:
        rec = rec.head(2)
        names = [f"{r['title']}" for _,r in rec.iterrows()]
        return f"\n\n**📚 You asked about {cat} 3 times! You may also like:** {', '.join(names)}"
    return ""