def library_info(query: str):
    q = query.lower().strip()

    # Small talk
    if q in ["hi","hello","hey","bye","good morning","good evening","thanks","thank you","ok","okay","hi there","hello there"]:
        return "Hi! I am IntelliLib Agent. I know everything about our library - 80 books, racks, availability and basic GK. How can I help you?"

    # Sunday
    if "sunday" in q:
        return "Library is closed on Sunday. Open from 9 AM to 6 PM, Monday to Saturday."

    # All timings variations + days
    if any(k in q for k in ["timing","timings","working hours","hours","open","close","closed","when"]):
        if "library" in q or "open" in q or "close" in q or "timing" in q or "hours" in q or "when" in q:
            return "Library is open from 9 AM to 6 PM, Monday to Saturday."

    if any(d in q for d in ["monday","tuesday","wednesday","thursday","friday","saturday"]):
        return "Yes, library open on that day: Mon-Sat 9 AM to 6 PM, Sunday closed. Max 3 books for 14 days."

    if "where" in q and "library" in q:
        return "Our library is located on the 2nd floor, Main Building. It is open from 9 AM to 6 PM, Monday to Saturday."

    if any(k in q for k in ["rule","rules","return","fine","issue"]):
        return "Books can be issued for 14 days. Late return fine is Rs. 10 per day. Library is open from 9 AM to 6 PM, Monday to Saturday."

    return None