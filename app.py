import streamlit as st
import json
import re
from google import genai
from google.genai import types
#import os
from bs4 import BeautifulSoup


#proxy = "http://localhost:24765"
#os.environ["HTTP_PROXY"] = proxy
#os.environ["HTTPS_PROXY"] = proxy


# --- Configuration ---
st.set_page_config(page_title="Fact-Checking Assistant", layout="wide")

# --- Gemini API Setup ---
systeminstruction = """
نقش: شما یک سیستم هوشمند و کامل برای تحلیل و صحت‌سنجی اخبار هستید. وظیفه شما اجرای یک فرآیند چهارمرحله‌ای بر روی متن ورودی است:
1.	استخراج ادعاهای قابل بررسی.
2.	تجزیه هر ادعا به سوالات بنیادین.
3.	تحقیق و گردآوری شواهد برای هر سوال.
4.	ارزیابی، جمع‌بندی و صدور رأی نهایی برای هر ادعا.
هدف: ارائه یک گزارش تحلیلی کامل در قالب یک آبجکت JSON که تمام مراحل صحت‌سنجی را برای هر ادعای موجود در متن ورودی مستند می‌کند.
دستورالعمل‌ها:
________________________________________
مرحله اول: شناسایی و استخراج ادعاها
•	متن خبر ورودی را به دقت مطالعه کن.
•	لیستی از تمام جملاتی که حاوی «ادعای قابل بررسی» هستند (شامل آمار، وعده، مقایسه، یا روابط علت و معلولی) را شناسایی کن.
•	از استخراج نظرات شخصی، جملات کلی یا سوالات خودداری کن.
________________________________________
مرحله دوم: تجزیه هر ادعا به سوالات اتمی
•	برای هر ادعایی که در مرحله اول استخراج کردی، آن را به اجزای منطقی و بنیادین خود تجزیه کن.
•	برای هر جزء، یک «سوال اتمی» دقیق و قابل جستجو طراحی کن که پاسخ به آن به درک درستیِ ادعای اصلی کمک کند.
________________________________________
مرحله سوم: تحقیق و گردآوری شواهد برای هر سوال اتمی
•	برای هر سوال اتمی که در مرحله دوم ساختی، یک فرآیند تحقیق کامل را اجرا کن:
o	طراحی عبارت‌های جستجو (Search Queries): چند عبارت کلیدی برای یافتن منابع موافق، مخالف و خنثی طراحی کن.
o	جمع‌آوری شواهد: با استفاده از عبارت‌های جستجو، حداقل ۳ و حداکثر ۵ شاهد معتبر از منابع دست اول (خبرگزاری‌ها، گزارش‌های رسمی، تحقیقات علمی) پیدا کن.
o	استخراج اطلاعات کلیدی: برای هر شاهد، اطلاعات زیر را استخراج کن:
..	date : تاریخ خبر به شمسی که در منبع ذکر شده است.
..	source_title : عنوان دقیق منبع.
..	quote : نقل قول مستقیم و مرتبط‌ترین بخش از متن منبع.
..	stance : موضع شاهد نسبت به «سوال اتمی» (نه ادعای اصلی). از برچسب‌های «موافق»، «مخالف» یا «خنثی/زمینه‌ای» استفاده کن.
..	interpretation : برداشت تحلیلی خودت از شاهد در یک یا دو جمله.
________________________________________
مرحله چهارم: ارزیابی و نتیجه‌گیری نهایی (مرحله جدید)
•	پس از جمع‌آوری شواهد برای تمام سوالات اتمیِ مرتبط با یک ادعا، یک تحلیل جامع برای ادعای اصلی ارائه بده. این تحلیل باید شامل موارد زیر باشد:
o	 summary_of_findings  : خلاصه‌ای چند جمله‌ای از مهم‌ترین شواهد موافق، مخالف و زمینه‌ای که به دست آوردی. این بخش باید تصویری کلی از چشم‌انداز اطلاعاتی پیرامون ادعا ارائه دهد.
o	verdict  : با توجه به مجموع شواهد، یکی از برچسب‌های زیر را برای «ادعای اصلی» انتخاب کن:
..	درست: تمام شواهد معتبر، ادعا را تایید می‌کنند.
..	نیمه درست: هسته اصلی ادعا درست است، اما جزئیات مهمی نادیده گرفته شده یا دقت کافی را ندارد.
..	گمراه‌کننده: ادعا ممکن است حاوی عناصری از حقیقت باشد، اما با حذف زمینه یا ارائه گزینشی اطلاعات، تصویر نادرستی را القا می‌کند.
..	نادرست: شواهد معتبر، ادعا را به وضوح رد می‌کنند.
..	غیر قابل بررسی: شواهد کافی و معتبری برای قضاوت در مورد درستی یا نادرستی ادعا در دسترس نیست.
o	reasoning  :به تفصیل توضیح بده که چرا این رأی را صادر کردی. به شواهد متناقض، نکات کلیدی، کیفیت منابع و زمینه‌های پنهان اشاره کن. این بخش باید به وضوح نشان دهد که چگونه شواهد جمع‌آوری‌شده در مرحله سوم به این نتیجه‌گیری منجر شده است.
________________________________________
نکات کلیدی
•	اگر در ادعا به تاریخ اشاره نشده، تاریخ روز را پیش‌فرض در نظر بگیر.
•	از گذاشتن citation مانند [2] در متن خروجی خودداری کن.
•	خروجی نهایی باید یک آبجکت JSON معتبر باشد.



-------

    "response_schema": {
        "type": "object",
        "properties": {
            "claims_and_evidences": {
                "type": "array",
                "description": "",
                "nullable": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "description": "",
                            "nullable": True,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claims_context": {
                                        "type": "string",
                                        "description": "",
                                        "nullable": True
                                    },
                                    "atomic_questions": {
                                        "type": "array",
                                        "description": "",
                                        "nullable": True,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "question": {
                                                    "type": "string",
                                                    "description": "",
                                                    "nullable": True
                                                },
                                                "evidences": {
                                                    "type": "array",
                                                    "description": "",
                                                    "nullable": True,
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "date": {
                                                                "type": "string",
                                                                "description": "",
                                                                "nullable": True
                                                            },
                                                            "source_title": {
                                                                "type": "string",
                                                                "description": "",
                                                                "nullable": True
                                                            },
                                                            "Quote": {
                                                                "type": "string",
                                                                "description": "",
                                                                "nullable": True
                                                            },
                                                            "stance": {
                                                                "type": "string",
                                                                "description": "",
                                                                "nullable": True
                                                            },
                                                            "interpretation": {
                                                                "type": "string",
                                                                "description": "",
                                                                "nullable": True
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "total_fact_checking": {
                "type": "array",
                "description": "",
                "nullable": True,
                "items": {
                    "type": "object",
                    "properties": {
                        
                        "summary_of_findings": {
                            "type": "string",
                            "description": "",
                            "nullable": True
                        },
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "درست",
                                "نیمه درست",
                                "گمراه کننده",
                                "نادرست",
                                "غیر قابل بررسی",
                            ]
                            "description": "",
                            "nullable": True
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "",
                            "nullable": True
                        }
                    }
                }
            }
        },
        "required": [
            "claims_and_evidences",
            "total_fact_checking"
            
        ]
    }
            
"""
apikey = "AIzaSyA0rz6G2sO3aqTiRpxt0VBcj5q6RHg3J3s"
client = genai.Client(api_key=apikey)
def callgemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
        "tools": [types.Tool(google_search=types.GoogleSearch())],
        "system_instruction": [types.Part.from_text(text=systeminstruction)],
        "temperature": 0.3,
        "thinking_config": types.ThinkingConfig(
            thinking_budget=0  # Disables thinking. Set a value in milliseconds to enable.
            )
    }
    )
    return response

def callgeminipro(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={
        "tools": [types.Tool(google_search=types.GoogleSearch())],
        "system_instruction": [types.Part.from_text(text=systeminstruction)],
        "temperature": 0.3 ,
    }
    )
    return response




# --- Utility Functions ---
def is_primitive(val):
    return isinstance(val, (str, int, float, bool)) or val is None

def build_table_from_dict(d):
    headers = list(d.keys())
    rows = []
    cells = []
    for k in headers:
        v = d[k]
        if is_primitive(v):
            cells.append(str(v))
        elif isinstance(v, dict):
            cells.append(build_table_from_dict(v))
        elif isinstance(v, list):
            cells.append(build_table_from_list(v))
        else:
            cells.append(str(v))
    rows.append(cells)
    return make_table(headers, rows)

def build_table_from_list(lst):
    if not lst:
        return "<table><tr><td></td></tr></table>"
    if all(is_primitive(item) for item in lst):
        rows = [[str(item)] for item in lst]
        return make_table(["value"], rows)

    headers = []
    seen = set()
    for item in lst:
        if isinstance(item, dict):
            for key in item.keys():
                if key not in seen:
                    headers.append(key)
                    seen.add(key)

    rows = []
    for item in lst:
        row = []
        for h in headers:
            val = item.get(h, "")
            if is_primitive(val):
                row.append(str(val))
            elif isinstance(val, dict):
                row.append(build_table_from_dict(val))
            elif isinstance(val, list):
                row.append(build_table_from_list(val))
            else:
                row.append(str(val))
        rows.append(row)
    return make_table(headers, rows)

def make_table(headers, rows):
    table = '<table border="1" style="border-collapse:collapse; width:100%;">'
    if headers:
        table += "<thead><tr>"
        for h in headers:
            table += f'<th style="border:1px solid #e0e0e0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight:bold; padding: 12px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">{h}</th>'
        table += "</tr></thead>"
    table += "<tbody>"
    for i, row in enumerate(rows):
        bg_color = "#fafafa" if i % 2 == 0 else "#ffffff"
        table += f'<tr style="background-color: {bg_color}; transition: all 0.3s ease;">'
        for cell in row:
            table += f'<td style="border:1px solid #e0e0e0; padding: 10px;">{cell}</td>'
        table += "</tr>"
    table += "</tbody></table>"
    return table

# --- Persistent state ---
if "results" not in st.session_state:
    st.session_state.results = None
if "status" not in st.session_state:
    st.session_state.status = None
if "response_obj" not in st.session_state:
    st.session_state.response_obj = None
if "pending" not in st.session_state:
    st.session_state.pending = False   # flag for rerun
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Gemini-2.5pro"
    
# --- Streamlit UI ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=B+Homa:wght@400;700&display=swap');
    
    * {
        font-family: 'B Homa', Tahoma, Arial, sans-serif !important;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'B Homa', Tahoma, Arial, sans-serif !important;
    }
    </style>
    
    <h1 style='direction:rtl; text-align:right; 
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-size: 52px;
               font-weight: 700;
               margin-bottom: 25px;
               text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
        🌐 دستیار راستی‌آزمایی 
    </h1>
    """,
    unsafe_allow_html=True
)

#  CSS
st.markdown(
    """
    <style>
    div.stRadio {
        direction: rtl;
        text-align: right;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 18px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    div.stRadio label {
        display: flex !important;
        flex-direction: row-reverse !important;
        align-items: center;
        justify-content: flex-end;
        font-family: 'B Homa', Tahoma, Arial, sans-serif !important;
        font-size: 19px;
        gap: 10px;
        color: white;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    div.stRadio input[type="radio"] {
        margin: 0 !important;
        transform: scale(1.3);
    }
    textarea {
        direction: rtl;
        text-align: right;
        font-family: 'B Homa', Tahoma, Arial, sans-serif !important;
        font-size: 17px;
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 18px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        transition: all 0.3s ease;
    }
    textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 0 20px rgba(118, 75, 162, 0.4);
        background: #ffffff;
    }
    div.stButton {
        direction: rtl;
        text-align: right;
        display: flex;
        justify-content: flex-start;
        margin-top: 15px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'B Homa', Tahoma, Arial, sans-serif !important;
        font-size: 20px;
        font-weight: 700;
        border: none;
        border-radius: 15px;
        padding: 14px 40px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    table { 
        font-family: 'B Homa', Arial, sans-serif !important;
        font-size: 16px; 
        direction: rtl; 
        text-align: right;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    th, td {
        border:1px solid #e0e0e0;
        padding: 12px;
    }
    th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    prompt = st.text_area(
        "📝 متن خبر یا اظهار برای راستی‌آزمایی را وارد کنید:",
        height=200,
        label_visibility="collapsed",  
        key="input_text",
        value=st.session_state.input_text

    )
    
    model_choice = st.radio(
        "💡 مدل را انتخاب کنید:",
        options=["Gemini-2.5pro", "Gemini-2.5flash"],
        index=0 if st.session_state.model_choice == "Gemini-2.5pro" else 1,
        horizontal=True,
        key="model_choice"  # <- This saves selection automatically
    )
    
    submit = st.button("✅ ارسال")

# ----------------- پردازش -----------------
# --- Status placeholder ---
if "status_placeholder" not in st.session_state:
    st.session_state.status_placeholder = st.empty()

# --- Handle submit ---
if submit:
    if not prompt.strip():
        st.session_state.status = """
        <div style='direction:rtl; text-align:right; 
                    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                    color:#c0392b; border:3px solid #e74c3c; 
                    padding:18px; border-radius:15px; margin-top:15px;
                    font-family: "B Homa", Tahoma, Arial, sans-serif;
                    font-size: 19px; font-weight: 700;
                    box-shadow: 0 6px 20px rgba(231, 76, 60, 0.3);'>
            ⚠️ لطفاً یک متن معتبر وارد کنید.
        </div>
        """
        st.session_state.results = None
        st.session_state.response_obj = None
        st.session_state.pending = False
        st.session_state.status_placeholder.markdown(st.session_state.status, unsafe_allow_html=True)

    else:
        # --- Step 1: show loading ---
        st.session_state.status = """
        <div style='direction:rtl; text-align:right; 
                    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                    color:#2980b9; border:3px solid #3498db; 
                    padding:18px; border-radius:15px; margin-top:15px;
                    font-family: "B Homa", Tahoma, Arial, sans-serif;
                    font-size: 19px; font-weight: 700;
                    box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
                    animation: pulse 2s infinite;'>
            ⏳ در حال تحلیل و راستی‌آزمایی...
        </div>
        <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        </style>
        """
        st.session_state.results = None
        st.session_state.response_obj = None
        st.session_state.pending = True
        st.session_state.status_placeholder.markdown(st.session_state.status, unsafe_allow_html=True)

# --- Step 2: run API if pending ---
if st.session_state.get("pending", False):
    import time
    time.sleep(0.1)  # short pause to let Streamlit render ⏳

    try:
        if model_choice == "Gemini-2.5flash":
            response = callgemini(prompt)
        else:
            response = callgeminipro(prompt)

        text = response.text
        st.session_state.results = text
        st.session_state.response_obj = response
        st.session_state.status = """
        <div style='direction:rtl; text-align:right; 
                    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                    color:#27ae60; border:3px solid #2ecc71; 
                    padding:18px; border-radius:15px; margin-top:15px;
                    font-family: "B Homa", Tahoma, Arial, sans-serif;
                    font-size: 19px; font-weight: 700;
                    box-shadow: 0 6px 20px rgba(46, 204, 113, 0.3);'>
            ✅ تحلیل با موفقیت انجام شد!
        </div>
        """
    except Exception as e:
        st.session_state.status = f"""
        <div style='direction:rtl; text-align:right; 
                    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                    color:#c0392b; border:3px solid #e74c3c; 
                    padding:18px; border-radius:15px; margin-top:15px;
                    font-family: "B Homa", Tahoma, Arial, sans-serif;
                    font-size: 19px; font-weight: 700;
                    box-shadow: 0 6px 20px rgba(231, 76, 60, 0.3);'>
            ❌ خطایی رخ داد: {e}
        </div>
        """
        st.session_state.results = None
        st.session_state.response_obj = None

    st.session_state.pending = False
    # overwrite the status placeholder
    st.session_state.status_placeholder.markdown(st.session_state.status, unsafe_allow_html=True)

# --- Always show latest state ---
if st.session_state.status:
    st.markdown(st.session_state.status, unsafe_allow_html=True)

if st.session_state.results:
    text = st.session_state.results
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)

            html = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=B+Homa:wght@400;700&display=swap');
            
            table { 
                font-family: 'B Homa', Arial, sans-serif !important;
                font-size: 16px; 
                direction: rtl; 
                text-align: right;
            }
            th, td {
                border:1px solid #e0e0e0;
                padding: 12px;
            }
            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 700;
            }
            </style>
            """
            html += build_table_from_dict(data)

            # --- Verdict, summary, reasoning ---
            if (
                "total_fact_checking" in data 
                and isinstance(data["total_fact_checking"], list) 
                and len(data["total_fact_checking"]) > 0
            ):
                fact_check = data["total_fact_checking"][0]
                summary_of_findings = fact_check.get("summary_of_findings", "")
                verdict = fact_check.get("verdict", "")
                reasoning = fact_check.get("reasoning", "")

                if verdict:
                    st.markdown(
                        f"""
                            <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                                        border:3px solid #fab1a0; 
                                        padding:18px; border-radius:15px; margin-bottom:15px;
                                        margin-top:25px;
                                        direction:rtl; text-align:right; 
                                        font-family: "B Homa", Tahoma, Arial, sans-serif;
                                        box-shadow: 0 6px 20px rgba(250, 177, 160, 0.4);' >
                                <b style="font-size:22px; color: #d63031;">🏷️ برچسب نهایی:</b><br>
                                <span style="font-size:20px; font-weight: 700; color: #e17055;">{verdict}</span>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )

                if summary_of_findings:
                    st.markdown(
                        f"""
                            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                                        border:3px solid #74b9ff; 
                                        padding:18px; border-radius:15px; margin-bottom:15px; 
                                        direction:rtl; text-align:right; 
                                        font-family: "B Homa", Tahoma, Arial, sans-serif;
                                        box-shadow: 0 6px 20px rgba(116, 185, 255, 0.4);' >
                                <b style="font-size:22px; color: #2980b9;">📊 نتیجه کلی راستی‌آزمایی:</b><br>
                                <span style="font-size:19px; color: #0984e3; line-height: 1.8;">{summary_of_findings}</span>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )

                if reasoning:
                    st.markdown(
                        f"""
                            <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                                        border:3px solid #fdcb6e; 
                                        padding:18px; border-radius:15px; margin-bottom:15px; 
                                        direction:rtl; text-align:right; 
                                        font-family: "B Homa", Tahoma, Arial, sans-serif;
                                        box-shadow: 0 6px 20px rgba(253, 203, 110, 0.4);' >
                                <b style="font-size:22px; color: #c0392b;">📚 استدلال:</b><br>
                                <span style="font-size:19px; color: #d63031; line-height: 1.8;">{reasoning}</span>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --- Extract references ---
                try:
                    if st.session_state.response_obj:
                        extract_ref = st.session_state.response_obj.candidates[0].grounding_metadata.search_entry_point.rendered_content
                        soup = BeautifulSoup(extract_ref, "html.parser")
                        chips = soup.select("div.carousel a.chip")
                        if chips:
                            chips_html = "<br>".join([f'• <a href="{chip.get("href")}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 600; transition: all 0.3s;">{chip.get_text(strip=True)}</a>' for chip in chips])
                            st.markdown(
                                f"""
                                <div style='background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); 
                                            border:3px solid #a29bfe; 
                                            padding:18px; border-radius:15px; margin-bottom:20px; 
                                            direction:rtl; text-align:right; 
                                            font-family: "B Homa", Tahoma, Arial, sans-serif;
                                            box-shadow: 0 6px 20px rgba(162, 155, 254, 0.4);' >
                                    <b style="font-size:22px; color: #6c5ce7;">🔎 پیشنهادات جستجو در گوگل:</b><br>
                                    <span style="font-size:19px; line-height: 2;">{chips_html}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                except Exception:
                    pass

            # --- Full table ---
            st.components.v1.html(html, height=4000, scrolling=True)

        except json.JSONDecodeError as e:
            st.markdown(
                f"""
                <div style='direction:rtl; text-align:right; 
                            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            color:#c0392b; border:3px solid #e74c3c; 
                            padding:18px; border-radius:15px; margin-top:15px;
                            font-family: "B Homa", Tahoma, Arial, sans-serif;
                            font-size: 19px; font-weight: 700;
                            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.3);'>
                    ❌ پاسخ سرویس دهنده ساختار مورد انتظار را ندارد: {e}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div style='direction:rtl; text-align:right; 
                        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                        color:#c0392b; border:3px solid #e74c3c; 
                        padding:18px; border-radius:15px; margin-top:15px;
                        font-family: "B Homa", Tahoma, Arial, sans-serif;
                        font-size: 19px; font-weight: 700;
                        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.3);'>
                ❌ پاسخ سرویس دهنده بدون ساختار میباشد.
            </div>
            """,
            unsafe_allow_html=True
        )
