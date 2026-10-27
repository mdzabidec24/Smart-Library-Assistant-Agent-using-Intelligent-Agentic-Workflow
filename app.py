import streamlit as st
import pandas as pd
import re
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from tools.book_search import search_books_core
from tools.recommend import recommend_books

st.set_page_config(page_title="IntelliLib Agent", page_icon="📚")
st.title("📚 IntelliLib - Library Agent")
st.markdown("<p style='font-size:17px;'>Hi! I am IntelliLib Agent here to assist you - find books, rack, copies, rules and info you need!</p>", unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2", temperature=0.2)
llm = get_llm()

def library_info(query: str):
    q = query.lower().strip()
    if q in ["hi","hello","hey","bye","good morning","good evening","thanks","thank you","ok","okay"]:
        return "Hi! I am IntelliLib Agent. I know everything about our library - 80 books, racks, availability and basic GK. How can I help you?"
    if "sunday" in q:
        return "Library is closed on Sunday. Open from 9 AM to 6 PM, Monday to Saturday."
    if any(k in q for k in ["timing","timings","working hours","hours","open","close","closed","when"]):
        if "library" in q or "open" in q or "close" in q or "timing" in q or "hours" in q or "when" in q:
            return "Library is open from 9 AM to 6 PM, Monday to Saturday."
    if any(d in q for d in ["monday","tuesday","wednesday","thursday","friday","saturday"]):
        return "Yes, library open on that day: Mon-Sat 9 AM to 6 PM, Sunday closed."
    if "where" in q and "library" in q:
        return "Our library is located on the 2nd floor, Main Building. It is open from 9 AM to 6 PM, Monday to Saturday."
    if any(k in q for k in ["rule","return","fine","issue"]):
        return "Books can be issued for 14 days. Late return fine is Rs. 10 per day. Library is open from 9 AM to 6 PM, Monday to Saturday."
    return None

class AgentState(TypedDict):
    current_query: str
    intent: str
    last_book: Optional[dict]
    last_records: Optional[List[dict]]
    last_answer: str
    answer: str

def is_gibberish_token(q):
    words = q.lower().split()
    for w in words:
        if len(w) >= 8 and w.isalpha():
            vowels = sum(1 for c in w if c in 'aeiou')
            if vowels <= 1:
                return True
            if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', w):
                return True
    clean = q.lower().replace(" ", "").replace("?","").replace(".","")
    if len(clean) >= 10 and clean.isalpha() and len(q.split()) == 1:
        vowels = sum(1 for c in clean if c in 'aeiou')
        if vowels <= 2:
            if re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', clean):
                return True
    return False

def contains_book_title(q, df):
    q_low = q.lower()
    for title in df['title'].astype(str).str.lower():
        if len(title) > 3 and title in q_low:
            return True
    return False

def router_node(state: AgentState):
    q_raw = state["current_query"].lower().strip()
    q = q_raw.replace("mcuh","much").replace("mucch","much").replace("muchh","much")
    q = q.replace("abouut","about").replace("bok","book").replace("boook","book")
    q = q.replace("gaems","games").replace("gaem","game").replace("developement","development")
    q = q.replace("copys","copies").replace("copie","copies").replace("rackk","rack")
    last_book = state.get("last_book")

    if last_book:
        if any(p in q for p in [" of it"," of this"," of that","where is it","where is this","where is that","copies of it","author of it","author of this","which rack","what rack","rack is it","rack of it","in which rack","in which rack is it","is it available","is this available","is that available","available now","is it there","is it in stock","how many copies","how much copies","copies available","how much copy","availability","available","cost is that","price is that","how much cost","how much is it","how much is that","much is it","much is that"]):
            if any(k in q for k in ["price","cost","mrp"]) and "copies" not in q:
                return {"intent": "book_price"}
            if "how much" in q and "copies" not in q and "copy" not in q:
                if "cost" in q or "price" in q or "is it" in q or "is that" in q:
                    return {"intent": "book_price"}
            if "much is it" in q or "much is that" in q:
                return {"intent": "book_price"}
            return {"intent": "pronoun"} if "price" not in q and "cost" not in q and "mrp" not in q else {"intent": "book_price"}
        if any(w in q for w in ["rack","copies","copy","available"]):
            if "it" in q.split() or "this" in q or "that" in q:
                return {"intent": "pronoun"}

    if any(p in q for p in [" of it"," of this"," of that","where is it","where is this","where is that","copies of it","author of it"]):
        if last_book:
            return {"intent": "pronoun"}
    if q in ["it","this","that","is it available now","available now","how many copies","how much copies","how much copies of it","how many copies of it","how much is it","how mcuh is it","much is it"]:
        if last_book:
            if "much" in q:
                return {"intent": "book_price"}
            return {"intent": "pronoun"}

    if is_gibberish_token(q_raw):
        return {"intent": "out_of_knowledge"}
    
    try:
        df_check = pd.read_csv("data/library.csv")
        has_title = contains_book_title(q, df_check)
        categories = [str(c).lower() for c in df_check['category'].unique()]
    except:
        has_title = False
        categories = []

    if q in ["hi","hello","hey","hai","hii","heya","yo","hola","namaste","bye","goodbye","see ya","tata","ok bye","good morning","good evening","thanks"]:
        return {"intent": "greeting"}
    if q.startswith(("hi ","hello ","hey ","hai ")) and len(q.split()) <= 2 and "book" not in q:
        return {"intent": "greeting"}
    if "manager" in q or "manger" in q or "librarian" in q:
        return {"intent": "manager"}
    if q in ["repeat","repeat that","say again","again"]:
        return {"intent": "repeat"}
    if any(k in q for k in ["price","cost","mrp"]) and "copies" not in q:
        return {"intent": "book_price"}
    if "how much" in q and "copies" not in q and "copy" not in q and ("cost" in q or "price" in q or "is it" in q or "is that" in q):
        return {"intent": "book_price"}
    if "much is it" in q or "much is that" in q:
        return {"intent": "book_price"} if last_book else {"intent": "out_of_knowledge"}
    if "fine" in q:
        return {"intent": "fine_rule"}
    if "payment" in q:
        return {"intent": "payment_rule"}
    if "issue" in q and ("rule" in q or "book" in q):
        return {"intent": "issue_rule"}
    if library_info(q) is not None:
        return {"intent": "library_info"}
    if any(w in q for w in ["stock","share","nifty","sensex"]):
        return {"intent": "general_knowledge_block"}
    if any(p in q for p in ["what are the books","books in library","what category","categories","category of books","categories of books","list of categories","how many categories","types of books","what books","categories of books are there"]):
        return {"intent": "count"}
    if "categor" in q:
        return {"intent": "count"}

    # === GK LIMITED TO SINGLE-WORD ONLY ===
    if q.startswith("what is") or q.startswith("what are"):
        if has_title:
            return {"intent": "search"}
        # that/by/of/about/related = book relation or 2 intents -> block
        if any(x in q for x in [" that ", " by ", " of ", " about ", " related to "]):
            if "book" in q:
                return {"intent": "search"}
            return {"intent": "out_of_knowledge"}
        if " and " in q:
            return {"intent": "out_of_knowledge"}
        cleaned = q.replace("what is","").replace("what are","").replace("?","").replace(".","").strip()
        words = [w for w in cleaned.split() if w not in ["the","a","an"]]
        if len(words) != 1:
            return {"intent": "out_of_knowledge"}
        single = words[0].lower()
        ALLOWED_GK = ["alien","rocket","country","apple","mango","banana","fruit","computer","laptop","pen","chair","table","school","orange","keyboard","mouse"]
        if single in ALLOWED_GK:
            return {"intent": "general_knowledge"}
        else:
            return {"intent": "out_of_knowledge"}

    if "colour of" in q or "color of" in q:
        return {"intent": "out_of_knowledge"}

    if "is there" in q and "book" in q:
        return {"intent": "search"}
    if any(phrase in q for phrase in ["book called","book named","books about","book about","books related to","book related to","about games","about development","development of games","about game"]):
        return {"intent": "search"}
    for cat in categories:
        if cat in q and len(cat) > 2 and cat not in ["knowledge"]:
            if "what is" not in q:
                return {"intent": "search"}
    if has_title:
        return {"intent": "search"}
    if "where is" in q or "where are" in q:
        if not has_title and "book" not in q:
            return {"intent": "out_of_knowledge"}
        else:
            return {"intent": "search"}
    if "who are you" in q or "what are you" in q:
        return {"intent": "who_are_you"}
    return {"intent": "out_of_knowledge"}

def greeting_node(state: AgentState):
    q = state["current_query"].lower()
    if "bye" in q:
        return {"answer": "Bye! Happy reading! 📚"}
    return {"answer": "Hi! Tell me which book you want? 📚"}
def repeat_node(state: AgentState):
    ans = state.get("last_answer", "")
    if ans:
        return {"answer": f"Repeating: {ans}"}
    if state.get("last_book"):
        b = state["last_book"]
        return {"answer": f"Repeating: '{b['title']}' - {b['available_copies']} copies at Rack {b['rack_no']}."}
    return {"answer": "Nothing to repeat yet. Ask a book!"}
def manager_node(state: AgentState):
    return {"answer": "Manager/librarian is at library counter in main block. Mon-Sat 9 AM - 6 PM."}
def who_are_you_node(state: AgentState):
    return {"answer": "I am IntelliLib Agent 📚 I help you find books, rack, copies & library rules. Tell me which book you need?"}
def out_of_knowledge_node(state: AgentState):
    return {"answer": "I don't know about that. I can help with books, rack, copies, and library rules. Tell me which book you need?"}
def general_knowledge_node(state: AgentState):
    q = state["current_query"]
    try:
        prompt = f"You are a 9th grade teacher. Explain '{q}' in 1 simple sentence, max 20 words, easy words. Do NOT mention library, rack, books, rules. Only simple GK."
        res = llm.invoke(prompt)
        desc = res.content if hasattr(res,'content') else str(res)
        desc = desc.strip().replace("\n"," ")
        sentences = re.split(r'(?<=[.!?])\s+', desc)
        desc = sentences[0]
        if len(desc) > 150:
            desc = desc[:150]
        return {"answer": f"{desc} 📚 For library books, tell me book name?"}
    except:
        return {"answer": "I don't know about that. I can help with books, rack, copies, and library rules."}
def general_knowledge_block_node(state: AgentState):
    return {"answer": "I can't give stock prices. I can help with books, rack, copies, library rules."}

def pronoun_node(state: AgentState):
    book = state.get("last_book")
    if not book:
        return {"answer": "Which book? Tell me book name."}
    try:
        df_fresh = pd.read_csv("data/library.csv")
        df_fresh['title'] = df_fresh['title'].astype(str).str.strip()
        df_fresh['rack_no'] = df_fresh['rack_no'].astype(str).str.strip()
        df_fresh['available_copies'] = pd.to_numeric(df_fresh['available_copies'], errors='coerce').fillna(0).astype(int)
        fresh = df_fresh[df_fresh['title'].str.lower() == str(book['title']).lower().strip()]
        if not fresh.empty:
            book = fresh.iloc[0].to_dict()
    except:
        pass
    copies = int(book.get('available_copies',0))
    q = state["current_query"].lower()
    try:
        df_for_rec = pd.read_csv("data/library.csv")
    except:
        df_for_rec = None
    rec = recommend_books(book['title'], df=df_for_rec, current_title=book['title'])
    if "available" in q or ("is it" in q and "there" in q) or ("is it" in q and "now" in q) or q in ["is it available","is it available now","is it available now?","available now"]:
        if copies == 0:
            return {"answer": f"Sorry, '{book['title']}' is not available now, will be coming soon! 📚{rec}"}
        else:
            return {"answer": f"Yes, '{book['title']}' is available - {copies} copies at Rack {book['rack_no']}.{rec}"}
    if "copies" in q or "copy" in q or "how many" in q or "how much" in q and "copies" in q:
        if copies == 0:
            return {"answer": f"Sorry, '{book['title']}' is not available now, will be coming soon! 📚{rec}"}
        return {"answer": f"'{book['title']}' has {copies} copies at Rack {book['rack_no']}.{rec}"}
    if copies == 0:
        return {"answer": f"Sorry, '{book['title']}' is not available now, will be coming soon! 📚{rec}"}
    if "author" in q:
        return {"answer": f"Author of '{book['title']}' is {book['author']} - {copies} copies at Rack {book['rack_no']}.{rec}"}
    return {"answer": f"'{book['title']}' is at Rack {book['rack_no']}, {copies} copies.{rec}"}

def fine_rule_node(state: AgentState):
    return {"answer": "Fine Rule: First 14 days free, then Rs. 5/day/book. Pay at counter. Contact manager."}
def payment_rule_node(state: AgentState):
    return {"answer": "Payment Rule: All payments at counter only. Contact manager."}
def issue_rule_node(state: AgentState):
    return {"answer": "Issue Rule: Max 3 books for 14 days. Need library card. Contact manager."}

def book_price_node(state: AgentState):
    msg = "Payment is done at billing section. Contact manager for price details."
    df = pd.read_csv("data/library.csv")
    for col in ['title','category','author','rack_no']:
        df[col] = df[col].astype(str).str.strip()
    df['available_copies'] = pd.to_numeric(df['available_copies'], errors='coerce').fillna(0).astype(int)
    result = search_books_core(state["current_query"], df, llm)
    if isinstance(result, list) and len(result)>0:
        book = result[0]
        try:
            fresh_row = df[df['title'].str.lower() == str(book['title']).lower().strip()]
            if not fresh_row.empty:
                book['rack_no'] = str(fresh_row.iloc[0]['rack_no']).strip()
                book['available_copies'] = int(fresh_row.iloc[0]['available_copies'])
        except:
            pass
        rec = recommend_books(book['title'], df=df, current_title=book['title'])
        return {"answer": f"'{book['title']}' - {book['available_copies']} copies at Rack {book['rack_no']}. {msg}{rec}", "last_book": book}
    if state.get("last_book"):
        b = state["last_book"]
        try:
            fresh = df[df['title'].str.lower() == str(b['title']).lower().strip()]
            if not fresh.empty:
                b['rack_no'] = str(fresh.iloc[0]['rack_no']).strip()
                b['available_copies'] = int(fresh.iloc[0]['available_copies'])
        except:
            pass
        rec = recommend_books(b['title'], df=df, current_title=b['title'])
        return {"answer": f"'{b['title']}' - {b['available_copies']} copies at Rack {b['rack_no']}. {msg}{rec}", "last_book": b}
    return {"answer": msg}

def library_info_node(state: AgentState):
    ans = library_info(state["current_query"])
    if ans:
        return {"answer": ans}
    return {"answer": "Library open Mon-Sat 9 AM - 6 PM, Sun closed. Rules: Max 3 books for 14 days. Contact manager at counter."}
def count_node(state: AgentState):
    df = pd.read_csv("data/library.csv")
    q = state["current_query"].lower()
    for cat in df['category'].unique():
        if str(cat).lower() in q:
            filt = df[df['category'].str.lower()==str(cat).lower()]
            titles = ", ".join([f"{r['title']} (Rack {r['rack_no']})" for _, r in filt.iterrows()])
            return {"answer": f"We have {len(filt)} {cat} books: {titles}"}
    cats = df['category'].unique()
    return {"answer": f"We have {len(cats)} categories: {', '.join(cats)}."}

def search_node(state: AgentState):
    df = pd.read_csv("data/library.csv")
    for col in ['title','category','author','rack_no']:
        df[col] = df[col].astype(str).str.strip()
    df['available_copies'] = pd.to_numeric(df['available_copies'], errors='coerce').fillna(0).astype(int)
    result = search_books_core(state["current_query"], df, llm)
    if isinstance(result, str) and "No books found" in result:
        q = state["current_query"].lower().replace("mcuh","much").replace("gaems","games").replace("gaem","game")
        tokens = [t for t in re.findall(r'\w+', q) if len(t)>2 and t not in ["there","book","books","about","related","there","is","are","the","for","of","to","development"]]
        if "game" in q or "games" in q:
            tokens.append("game")
        if "development" in q or "develop" in q:
            tokens.append("game")
            tokens.append("development")
        if tokens:
            mask = pd.Series([False]*len(df))
            for tok in tokens:
                base = tok.rstrip('s')
                mask = mask | df['title'].str.lower().str.contains(base, na=False)
                mask = mask | df['category'].str.lower().str.contains(base, na=False)
            matched = df[mask]
            if not matched.empty:
                result = matched.to_dict(orient='records')
    if isinstance(result, str) and "No books found" in result:
        return {"answer": f"Sorry, that book not found now, will be coming soon! We have: {', '.join(df['category'].unique())}.", "last_book": None}
    if isinstance(result, list) and len(result)>0:
        book = result[0]
        try:
            fresh_row = df[df['title'].str.lower() == str(book['title']).lower().strip()]
            if not fresh_row.empty:
                book['rack_no'] = str(fresh_row.iloc[0]['rack_no']).strip()
                book['available_copies'] = int(fresh_row.iloc[0]['available_copies'])
        except:
            pass
        if int(book['available_copies']) == 0:
            return {"answer": f"Sorry, '{book['title']}' is not available now, will be coming soon! 📚", "last_book": book}
        try:
            topic = book['title'].replace("'","")
            desc_prompt = f"What is {topic}? Explain in 1 simple sentence for 9th grade kid. Max 20 words."
            res = llm.invoke(desc_prompt)
            desc = res.content if hasattr(res,'content') else str(res)
            desc = desc.strip().replace("\n"," ")
            sentences = re.split(r'(?<=[.!?])\s+', desc)
            desc = sentences[0] if sentences else desc
            if len(desc) > 160:
                desc = desc[:160]
        except:
            desc = f"It is about learning {book['title']} basics in simple way."
        if len(result)==1:
            rec = recommend_books(state["current_query"], df, current_title=book['title'])
            ans = f"'{book['title']}' by {book['author']} - {book['available_copies']} copies at Rack {book['rack_no']}. About: {desc}{rec}"
        else:
            titles = ", ".join([f"{r['title']} (Rack {r['rack_no']})" for r in result])
            ans = f"Found {len(result)} books: {titles}"
        return {"answer": ans, "last_book": book, "last_records": result}
    return {"answer": "Tell me book name?"}

graph = StateGraph(AgentState)
for name, func in [("router", router_node), ("greeting", greeting_node), ("pronoun", pronoun_node), 
                   ("fine_rule", fine_rule_node), ("payment_rule", payment_rule_node),
                   ("issue_rule", issue_rule_node), ("book_price", book_price_node),
                   ("library_info", library_info_node), ("count", count_node), ("search", search_node),
                   ("repeat", repeat_node), ("manager", manager_node), ("out_of_knowledge", out_of_knowledge_node),
                   ("general_knowledge", general_knowledge_node),
                   ("general_knowledge_block", general_knowledge_block_node),
                   ("who_are_you", who_are_you_node)]:
    graph.add_node(name, func)

def route_decision(state):
    m = {"greeting": "greeting", "repeat": "repeat", "manager": "manager","pronoun": "pronoun", "fine_rule": "fine_rule","payment_rule": "payment_rule", "issue_rule": "issue_rule","book_price": "book_price", "library_info": "library_info","count": "count", "search": "search", "out_of_knowledge": "out_of_knowledge","general_knowledge": "general_knowledge","general_knowledge_block": "general_knowledge_block","who_are_you": "who_are_you"}
    return m.get(state["intent"], "out_of_knowledge")

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision, {"greeting": "greeting", "pronoun": "pronoun", "fine_rule": "fine_rule","payment_rule": "payment_rule", "issue_rule": "issue_rule","book_price": "book_price", "library_info": "library_info","count": "count", "search": "search", "repeat": "repeat", "manager": "manager", "out_of_knowledge": "out_of_knowledge","general_knowledge": "general_knowledge","general_knowledge_block": "general_knowledge_block","who_are_you": "who_are_you"})
for n in ["greeting","pronoun","fine_rule","payment_rule","issue_rule","book_price","library_info","count","search","repeat","manager","out_of_knowledge","general_knowledge","general_knowledge_block","who_are_you"]:
    graph.add_edge(n, END)

app_graph = graph.compile()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'graph_state' not in st.session_state:
    st.session_state.graph_state = {"last_book": None, "last_answer": ""}

def add_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(f"<p style='font-size:17px;'>{msg['content']}</p>", unsafe_allow_html=True)

prompt = st.chat_input("Ask about books...")

if prompt:
    with st.chat_message("user"):
        st.markdown(f"<p style='font-size:17px;'>{prompt}</p>", unsafe_allow_html=True)
    add_history("user", prompt)
    result = app_graph.invoke({"current_query": prompt,"intent": "","last_book": st.session_state.graph_state.get("last_book"),"last_records": None,"last_answer": st.session_state.graph_state.get("last_answer"),"answer": ""})
    ans = result["answer"]
    if result.get("last_book"):
        st.session_state.graph_state["last_book"] = result["last_book"]
    st.session_state.graph_state["last_answer"] = ans
    with st.chat_message("assistant"):
        st.markdown(f"<p style='font-size:17px;'>{ans}</p>", unsafe_allow_html=True)
    add_history("assistant", ans)