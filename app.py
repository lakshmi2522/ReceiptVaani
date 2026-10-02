
import json
import os
import time
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="ReceiptVaani", page_icon="🧾", layout="wide")

# -------------------- UI --------------------
st.markdown("""
<style>
:root{--bg:#071018;--panel:#0e1823;--line:#263545;--text:#f3f7f8;--muted:#91a1ad;--cyan:#48d5e7;--lime:#b1eb5c;--amber:#f6b14e;--red:#f46366}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 85% 0%,#123043 0%,#071018 43%);color:var(--text)}
.block-container{max-width:1180px;padding-top:1.8rem}
h1{font-size:3.7rem!important;letter-spacing:-.06em}
h1 span{color:var(--cyan)}
.card{background:rgba(14,24,35,.9);border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.2rem;margin:.7rem 0}
.metric{background:#101c28;border:1px solid var(--line);border-radius:14px;padding:.95rem}
.metric .sm{color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.1em}
.metric .big{color:var(--text);font-size:1.45rem;font-weight:800;margin-top:.25rem}
.badge{display:inline-block;padding:.22rem .5rem;border-radius:999px;background:#12202e;color:var(--cyan);font-size:.72rem;font-weight:800}
.alert{background:#302413;border:1px solid #8d682c;border-radius:14px;padding:.9rem}
.ok{background:#10261e;border:1px solid #29533f;border-radius:14px;padding:.9rem}
.evidence{background:#0d1721;border-left:4px solid var(--cyan);padding:.7rem .85rem;border-radius:8px;margin:.45rem 0}
.muted{color:var(--muted)}
[data-testid="stFileUploader"] section{border-color:var(--line)}
button[kind="primary"]{background:var(--cyan)!important;color:#061017!important}
</style>
""", unsafe_allow_html=True)

# -------------------- Schemas --------------------
class Item(BaseModel):
    description: str
    amount: float | None = None
    category: str | None = None
    raw_text: str | None = None

class Bill(BaseModel):
    vendor: str | None = None
    bill_date: str | None = None
    due_date: str | None = None
    currency: str = "INR"
    subtotal: float | None = None
    tax_total: float | None = None
    total: float | None = None
    items: list[Item] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

# -------------------- Demo data --------------------
PREV = Bill(
    vendor="Demo Power",
    bill_date="2026-08-25",
    currency="INR",
    subtotal=1050,
    tax_total=190,
    total=1240,
    items=[
        Item(description="Energy charge", amount=1050, category="usage", raw_text="Energy charge ₹1,050"),
        Item(description="GST", amount=190, category="tax", raw_text="GST ₹190"),
    ],
)
CUR = Bill(
    vendor="Demo Power",
    bill_date="2026-09-25",
    currency="INR",
    subtotal=1648,
    tax_total=201,
    total=1849,
    items=[
        Item(description="Energy charge", amount=1120, category="usage", raw_text="Energy charge ₹1,120"),
        Item(description="GST", amount=201, category="tax", raw_text="GST ₹201"),
        Item(description="Service fee", amount=528, category="fee", raw_text="Service fee ₹528"),
    ],
)

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, " ".join(a.lower().split()), " ".join(b.lower().split())).ratio()

def compare(cur: Bill, prev: Bill):
    anomalies = []
    total_change = None

    if cur.total is not None and prev.total not in (None, 0):
        total_change = round((cur.total - prev.total) / prev.total * 100, 2)
        if total_change >= 15:
            anomalies.append({
                "type": "total_increase",
                "severity": "high" if total_change >= 35 else "medium",
                "item": "Bill total",
                "description": "The current bill total increased materially versus the previous bill.",
                "current_amount": cur.total,
                "previous_amount": prev.total,
                "delta_percent": total_change,
                "evidence": f"Previous total ₹{prev.total:,.2f} → current total ₹{cur.total:,.2f}",
            })

    for item in cur.items:
        if item.amount is None:
            continue
        best, score = None, 0.0
        for old in prev.items:
            s = sim(item.description, old.description)
            if s > score:
                best, score = old, s

        if best is None or score < 0.68:
            anomalies.append({
                "type": "new_line_item",
                "severity": "high" if item.category == "fee" else "medium",
                "item": item.description,
                "description": "This charge has no close match in the previous bill.",
                "current_amount": item.amount,
                "previous_amount": None,
                "delta_percent": None,
                "evidence": item.raw_text or item.description,
            })
        elif best.amount not in (None, 0):
            delta = round((item.amount - best.amount) / best.amount * 100, 2)
            if delta >= 20:
                anomalies.append({
                    "type": "line_item_increase",
                    "severity": "high" if delta >= 50 else "medium",
                    "item": item.description,
                    "description": "This matched charge increased materially versus the previous bill.",
                    "current_amount": item.amount,
                    "previous_amount": best.amount,
                    "delta_percent": delta,
                    "evidence": item.raw_text or item.description,
                })

    return {
        "current_total": cur.total,
        "previous_total": prev.total,
        "total_change_percent": total_change,
        "anomalies": anomalies,
    }

def get_key():
    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret = ""
    return st.session_state.get("manual_key") or os.getenv("GEMINI_API_KEY") or secret

def extract_bill(uploaded_file):
    key = get_key()
    if not key:
        raise RuntimeError("Add a Gemini API key in the sidebar or use Demo Mode.")
    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    prompt = """
You are ReceiptVaani's bill extraction engine.
Extract only information visibly supported by the bill/receipt.
Never invent values. Use null when unclear.
Return JSON matching the schema.
Extract vendor, bill date, due date, currency, subtotal, tax_total, total and EVERY visible monetary line item.
For every line item, include a short exact raw_text snippet when possible.
Category should be usage, tax, fee, discount, subscription, product, service, or other.
Put ambiguities in notes.
"""
    part = types.Part.from_bytes(
        data=uploaded_file.getvalue(),
        mime_type=uploaded_file.type,
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt, part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Bill,
            temperature=0,
        ),
    )
    if not response.text:
        raise RuntimeError("No result returned by Gemini.")
    return Bill.model_validate_json(response.text)

def answer_question(cur: Bill, comp: dict, question: str, language: str):
    key = get_key()
    if not key:
        # Deterministic fallback so Demo Mode never breaks.
        new_items = [a for a in comp.get("anomalies", []) if a["type"] == "new_line_item"]
        if language == "Kannada" and new_items:
            a = new_items[0]
            return f"ಈ ಬಿಲ್‌ನಲ್ಲಿ {a['item']} ಎಂಬ ಹೊಸ charge ₹{a['current_amount']:.0f} ಇದೆ. ಇದು ಹಿಂದಿನ ಬಿಲ್‌ನಲ್ಲಿ ಕಾಣಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು verify ಮಾಡಿ."
        if language == "Hindi" and new_items:
            a = new_items[0]
            return f"इस बिल में {a['item']} का नया charge ₹{a['current_amount']:.0f} है। यह पिछले बिल में नहीं था। कृपया इसे verify करें।"
        if new_items:
            a = new_items[0]
            return f"A new {a['item']} of ₹{a['current_amount']:.0f} appears in the current bill and was not present in the previous bill. Please verify it."
        return "I need a previous bill or more evidence to explain the change."

    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    payload = json.dumps({"current_bill": cur.model_dump(), "comparison": comp})
    prompt = f"""
You are ReceiptVaani. Answer the user's question using ONLY the supplied bill/comparison evidence.
Never invent a reason for a charge. If evidence is insufficient, explicitly say so.
Never call an unusual charge fraud. Say it is unusual based on available history and advise verification.
Answer in {language}, in 2-4 short sentences.

Evidence:
{payload}

Question:
{question}
"""
    r = client.models.generate_content(model=model, contents=prompt)
    return (r.text or "").strip()

def evaluation():
    # Small controlled benchmark used to demonstrate how the rules are evaluated.
    cases=[]
    for i in range(24):
        p=1000+i*10
        prev=[Item(description="Base plan",amount=600)]
        cur=[Item(description="Base plan",amount=600)]
        label=False
        if i%4==0:
            cur.append(Item(description="Service fee",amount=250)); label=True
        elif i%4==1:
            cur=[Item(description="Base plan",amount=780)]; label=True
        elif i%4==2:
            pass
        else:
            cur.append(Item(description="GST",amount=180)); prev.append(Item(description="GST",amount=180))
        prev_bill=Bill(total=sum(x.amount or 0 for x in prev),items=prev)
        cur_bill=Bill(total=sum(x.amount or 0 for x in cur),items=cur)
        result=compare(cur_bill,prev_bill)
        pred=any(x["type"] in {"new_line_item","line_item_increase"} for x in result["anomalies"])
        cases.append((label,pred))
    tp=sum(y and p for y,p in cases); fp=sum((not y) and p for y,p in cases)
    fn=sum(y and (not p) for y,p in cases); tn=sum((not y) and (not p) for y,p in cases)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {"N":len(cases),"TP":tp,"FP":fp,"FN":fn,"TN":tn,"Precision":precision,"Recall":recall,"F1":f1}

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🧾 ReceiptVaani")
    mode=st.radio("Mode",["Demo Mode","Live AI"],index=0)
    language=st.selectbox("Explain in",["English","Kannada","Hindi"])
    if mode=="Live AI":
        st.session_state["manual_key"]=st.text_input("Gemini API key",type="password",value="")
        st.caption("The key is kept in this session only.")
    st.divider()
    st.caption("AI advises. The user decides.")
    st.caption("Do not call an unusual charge fraud from the bill alone.")

st.markdown('<span class="badge">BFWAI / HACK 26 · PS-06 · FINAL MVP</span>', unsafe_allow_html=True)
st.title("Receipt<span>Vaani</span>", unsafe_allow_html=True)
st.markdown("### Understand every rupee, in your own language.")
st.markdown('<p class="muted">Read → Compare → Flag → Explain → User decides</p>', unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(["🔍 Analyze","🧠 Ask","📊 Evaluation","🛡️ Trust"])

# -------------------- Analyze --------------------
with tab1:
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("Current bill")
        current=st.file_uploader("Upload current bill",type=["jpg","jpeg","png","pdf"],key="cur")
        if current: st.caption(current.name)
        st.markdown("</div>",unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("Previous bill")
        previous=st.file_uploader("Upload previous bill",type=["jpg","jpeg","png","pdf"],key="prev")
        if previous: st.caption(previous.name)
        st.markdown("</div>",unsafe_allow_html=True)

    if st.button("Analyze",type="primary",use_container_width=True):
        start=time.perf_counter()
        try:
            if mode=="Demo Mode":
                cur,prev=CUR,PREV
            else:
                if not current:
                    st.error("Upload the current bill.")
                    st.stop()
                cur=extract_bill(current)
                prev=extract_bill(previous) if previous else Bill()
            comp=compare(cur,prev) if prev.items else {"current_total":cur.total,"previous_total":None,"total_change_percent":None,"anomalies":[]}
            st.session_state["cur"]=cur.model_dump()
            st.session_state["prev"]=prev.model_dump()
            st.session_state["comp"]=comp
            st.session_state["elapsed"]=round(time.perf_counter()-start,2)
            st.success("Analysis complete.")
        except Exception as e:
            st.error(str(e))

    curd=st.session_state.get("cur")
    prevd=st.session_state.get("prev")
    comp=st.session_state.get("comp")
    if curd:
        cols=st.columns(4)
        vals=[
            ("Vendor",curd.get("vendor") or "—"),
            ("Current",f"₹{curd.get('total') or 0:,.2f}"),
            ("Previous",f"₹{prevd.get('total'):,.2f}" if prevd and prevd.get("total") else "—"),
            ("Change",f"{comp.get('total_change_percent'):.1f}%" if comp.get("total_change_percent") is not None else "—")
        ]
        for c,(lab,val) in zip(cols,vals):
            with c:
                st.markdown(f'<div class="metric"><div class="sm">{lab}</div><div class="big">{val}</div></div>',unsafe_allow_html=True)

        l,r=st.columns([1.45,1])
        with l:
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.subheader("Charges")
            rows=[{"Charge":i["description"],"Amount":f"₹{i['amount']:,.2f}" if i.get("amount") is not None else "—"} for i in curd.get("items",[])]
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.markdown("</div>",unsafe_allow_html=True)
        with r:
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.subheader("Unusual changes")
            anomalies=comp.get("anomalies",[])
            if not anomalies:
                st.markdown('<div class="ok">✓ No unusual change flagged by the current rules.</div>',unsafe_allow_html=True)
            for a in anomalies:
                pct=f"<br><b>Change:</b> +{a['delta_percent']}%" if a.get("delta_percent") is not None else ""
                st.markdown(f'<div class="alert"><b>⚠ {a["item"]}</b><br>{a["description"]}{pct}<br><br><b>Evidence:</b> {a["evidence"]}</div>',unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)

        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("Evidence Lens")
        for a in comp.get("anomalies",[]):
            st.markdown(f'<div class="evidence"><b>Flagged:</b> {a["item"]}<br><b>Why:</b> {a["description"]}<br><b>Source:</b> {a["evidence"]}</div>',unsafe_allow_html=True)
        st.caption("Trace = current line item + previous-bill comparison. The system does not auto-dispute.")
        st.markdown("</div>",unsafe_allow_html=True)

# -------------------- Ask --------------------
with tab2:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Ask ReceiptVaani")
    q=st.text_input("Question",value="Why is my bill higher?")
    if st.button("Explain",type="primary"):
        if not st.session_state.get("cur") or not st.session_state.get("comp"):
            st.warning("Analyze the bill first.")
        else:
            try:
                ans=answer_question(Bill.model_validate(st.session_state["cur"]),st.session_state["comp"],q,language)
                st.markdown(f'<div class="ok"><b>{language}</b><br><br>{ans}</div>',unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))
    st.markdown("</div>",unsafe_allow_html=True)

# -------------------- Evaluation --------------------
with tab3:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Controlled anomaly evaluation")
    st.write("Synthetic engineering benchmark for the deterministic comparison rules. Replace/expand this with your final evaluation set before claiming results.")
    if st.button("Run evaluation",type="primary"):
        st.session_state["eval"]=evaluation()
    ev=st.session_state.get("eval")
    if ev:
        cols=st.columns(4)
        for c,(lab,val) in zip(cols,[("Cases",ev["N"]),("Precision",f"{ev['Precision']*100:.1f}%"),("Recall",f"{ev['Recall']*100:.1f}%"),("F1",f"{ev['F1']*100:.1f}%")]):
            with c:
                st.markdown(f'<div class="metric"><div class="sm">{lab}</div><div class="big">{val}</div></div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"TP":ev["TP"],"FP":ev["FP"],"FN":ev["FN"],"TN":ev["TN"]}]),hide_index=True,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

# -------------------- Trust --------------------
with tab4:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Human-in-the-loop")
    st.markdown("**AI can flag. AI cannot accuse.**")
    st.write("The system describes a charge as unusual based on available bill history. It does not automatically label it fraud or submit a dispute.")
    c1,c2=st.columns(2)
    with c1: st.button("✓ Verify this charge",use_container_width=True)
    with c2: st.button("Ignore for now",use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Failure handling")
    st.warning("If the image is unclear or there is no comparison history, ReceiptVaani should return uncertainty rather than inventing a reason.")
    st.code("Insufficient evidence to explain this charge. Upload a clearer bill or an older bill.")
    st.markdown("</div>",unsafe_allow_html=True)

if "elapsed" in st.session_state:
    st.caption(f"Latest analysis time: {st.session_state['elapsed']} s")
