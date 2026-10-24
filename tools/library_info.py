def library_info(query: str):
    q = query.lower().strip()
    # Small talk
    if q in ["hi","hello","hey","bye","good morning","good evening","thanks","thank you","ok","okay"]:
        return "Hi! I am IntelliLib Agent. I know everything about our library - 80 books, racks, availability and basic GK. How can I help you?"
    if "sunday" in q:
        return "Library is closed on Sunday. Open from 9 AM to 6 PM, Monday to Saturday."
    if any(k in q for k in ["timing","working hours","hours","open","close"]):
        return "Library is open from 9 AM to 6 PM, Monday to Saturday."
    if "where" in q and "library" in q:
        return "Our library is located on the 2nd floor, Main Building. It is open from 9 AM to 6 PM, Monday to Saturday."
    if any(k in q for k in ["rule","return","fine"]):
        return "Books can be issued for 14 days. Late return fine is Rs. 10 per day. Library is open from 9 AM to 6 PM, Monday to Saturday."
    return None