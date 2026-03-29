"""
question_generator.py
=====================
Generates role-specific MCQ and interview questions.

Priority chain:
  1. FLAN-T5 (local HuggingFace model — zero cost, offline)
  2. Groq API free tier (LLaMA-3 — 14,400 req/day free)
  3. Curated static question bank (always works, zero dependencies)

Install once:
  pip install transformers torch  # for FLAN-T5
  pip install groq                # optional, for Groq fallback
"""

from __future__ import annotations
import os
import json
import random
import re
from typing import Optional

# ── Static question bank ──────────────────────────────────────────────────────
# Comprehensive curated questions per role — used as primary or fallback.

STATIC_MCQ: dict[str, list[dict]] = {
    "DATA-SCIENCE": [
        {"q": "Which library is primarily used for data manipulation in Python?",
         "options": ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"], "answer": "Pandas",
         "explanation": "Pandas provides DataFrame and Series structures ideal for data manipulation, cleaning, and analysis. NumPy handles numerical arrays, Matplotlib handles plotting, and Scikit-learn handles ML models."},
        {"q": "What does 'overfitting' mean in machine learning?",
         "options": ["Model performs well on training, poorly on test data",
                     "Model performs poorly on both datasets",
                     "Model has too few parameters",
                     "Model trains too slowly"], "answer": "Model performs well on training, poorly on test data",
         "explanation": "Overfitting occurs when a model memorizes training data including noise, causing it to perform well on training data but poorly on unseen test data. Regularization and cross-validation help prevent it."},
        {"q": "Which algorithm is used for classification tasks?",
         "options": ["Linear Regression", "K-Means", "Random Forest", "PCA"], "answer": "Random Forest",
         "explanation": "Random Forest is an ensemble classification algorithm. Linear Regression predicts continuous values, K-Means is for clustering (unsupervised), and PCA is for dimensionality reduction."},
        {"q": "What is the purpose of cross-validation?",
         "options": ["Speed up training", "Reduce dataset size",
                     "Evaluate model generalization", "Increase accuracy always"], "answer": "Evaluate model generalization",
         "explanation": "Cross-validation (e.g. k-fold) splits data into multiple train/test sets to reliably estimate how well a model generalizes to unseen data, reducing the risk of overfitting to a single test split."},
        {"q": "Which metric is best for imbalanced classification datasets?",
         "options": ["Accuracy", "F1-Score", "Mean Squared Error", "R²"], "answer": "F1-Score",
         "explanation": "F1-Score balances Precision and Recall, making it ideal for imbalanced datasets where accuracy can be misleading (e.g. 95% accuracy by always predicting the majority class). MSE and R² are regression metrics."},
        {"q": "What does PCA stand for?",
         "options": ["Principal Component Analysis", "Predicted Class Algorithm",
                     "Polynomial Curve Approximation", "Probabilistic Cluster Assignment"],
         "answer": "Principal Component Analysis",
         "explanation": "PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms features into orthogonal principal components ranked by variance explained, reducing complexity while preserving information."},
        {"q": "Which activation function is commonly used in output layer for binary classification?",
         "options": ["ReLU", "Sigmoid", "Tanh", "Softmax"], "answer": "Sigmoid",
         "explanation": "Sigmoid squashes output to [0,1], making it perfect for binary classification probability. ReLU is used in hidden layers, Softmax for multi-class output, and Tanh outputs [-1,1]."},
        {"q": "What is a hyperparameter?",
         "options": ["Parameter learned from training data",
                     "Configuration set before training begins",
                     "Output of the model",
                     "Feature in the dataset"], "answer": "Configuration set before training begins",
         "explanation": "Hyperparameters (e.g. learning rate, max depth, number of trees) are set before training and control the model structure. Model parameters (weights, biases) are learned from data during training."},
        {"q": "Which SQL clause filters aggregated results?",
         "options": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"], "answer": "HAVING",
         "explanation": "HAVING filters groups after aggregation (e.g. HAVING COUNT(*) > 5). WHERE filters individual rows before aggregation. GROUP BY groups rows. ORDER BY sorts results."},
        {"q": "What is the bias-variance tradeoff?",
         "options": ["Tradeoff between model complexity and generalization",
                     "Tradeoff between CPU and GPU usage",
                     "Tradeoff between accuracy and speed",
                     "Tradeoff between features and labels"],
         "answer": "Tradeoff between model complexity and generalization",
         "explanation": "High bias = underfitting (too simple model). High variance = overfitting (too complex model). The tradeoff involves finding the right complexity that minimizes both bias and variance for best generalization."},
    ],
    "WEB-DEVELOPER": [
        {"q": "Which HTML tag is used to link an external CSS file?",
         "options": ["<style>", "<script>", "<link>", "<css>"], "answer": "<link>",
         "explanation": "<link rel='stylesheet' href='style.css'> connects external CSS. <style> is for inline CSS within HTML. <script> is for JavaScript. There is no <css> tag."},
        {"q": "What does CSS 'box model' include?",
         "options": ["Content, Padding, Border, Margin",
                     "Width, Height, Color, Font",
                     "Display, Position, Float, Clear",
                     "Background, Border, Shadow, Opacity"],
         "answer": "Content, Padding, Border, Margin",
         "explanation": "The CSS box model describes how every element is a box with: Content (inner text/media), Padding (space inside border), Border (the edge), and Margin (space outside border). Understanding this is critical for layout."},
        {"q": "Which JavaScript method removes the last element from an array?",
         "options": ["shift()", "pop()", "splice()", "slice()"], "answer": "pop()",
         "explanation": "pop() removes and returns the last array element. shift() removes the first. splice() removes at a specific index. slice() creates a new array without modifying the original."},
        {"q": "What does REST stand for?",
         "options": ["Representational State Transfer", "Remote Execution Service Technology",
                     "Resource Enabled Server Transfer", "Reactive Event Streaming Technology"],
         "answer": "Representational State Transfer",
         "explanation": "REST (Representational State Transfer) is an architectural style for APIs using HTTP methods (GET, POST, PUT, DELETE) and stateless communication with resources identified by URLs."},
        {"q": "Which HTTP status code means 'Not Found'?",
         "options": ["200", "301", "404", "500"], "answer": "404",
         "explanation": "404 = Not Found. 200 = OK (success). 301 = Moved Permanently (redirect). 500 = Internal Server Error. 401 = Unauthorized. 403 = Forbidden."},
        {"q": "In React, what is the purpose of useState?",
         "options": ["Fetch API data", "Manage component state",
                     "Route between pages", "Style components"], "answer": "Manage component state",
         "explanation": "useState is a React Hook that lets functional components hold and update local state. It returns [currentValue, setterFunction]. State changes trigger a re-render of the component."},
        {"q": "Which CSS property controls the stacking order of elements?",
         "options": ["position", "display", "z-index", "overflow"], "answer": "z-index",
         "explanation": "z-index controls the vertical stacking order of positioned elements (those with position: relative/absolute/fixed/sticky). Higher z-index = in front. Only works on positioned elements."},
        {"q": "What is the purpose of async/await in JavaScript?",
         "options": ["Handle synchronous code", "Handle asynchronous operations more cleanly",
                     "Create new threads", "Optimize memory usage"],
         "answer": "Handle asynchronous operations more cleanly",
         "explanation": "async/await is syntactic sugar over Promises, making asynchronous code look synchronous and easier to read/debug. JavaScript is single-threaded — async/await doesn't create threads, it manages the event loop."},
        {"q": "Which database type stores data as key-value pairs?",
         "options": ["Relational", "Graph", "NoSQL/Document", "NoSQL/Key-Value"],
         "answer": "NoSQL/Key-Value",
         "explanation": "Key-Value stores (Redis, DynamoDB) store data as simple key→value pairs for fast lookup. Relational DBs use tables. Document DBs (MongoDB) store JSON-like documents. Graph DBs store nodes and relationships."},
        {"q": "What does 'responsive design' mean?",
         "options": ["Fast page load", "Design adapts to different screen sizes",
                     "Interactive animations", "Server-side rendering"],
         "answer": "Design adapts to different screen sizes",
         "explanation": "Responsive design uses CSS media queries, flexible grids, and fluid images so a website looks and works well on all screen sizes — desktop, tablet, and mobile — without separate versions."},
    ],
    "HR": [
        {"q": "What does ATS stand for in recruitment?",
         "options": ["Applicant Tracking System", "Automated Testing Software",
                     "Annual Talent Survey", "Assessment Tracking Suite"],
         "answer": "Applicant Tracking System",
         "explanation": "ATS (Applicant Tracking System) software helps HR teams manage job applications, parse resumes, filter candidates by keywords, and track candidates through the hiring pipeline. Examples: Workday, Taleo, Greenhouse."},
        {"q": "What is 'onboarding' in HR?",
         "options": ["Employee resignation process",
                     "Process of integrating new employees",
                     "Annual performance review",
                     "Salary negotiation"], "answer": "Process of integrating new employees",
         "explanation": "Onboarding is the structured process of integrating new hires — covering orientation, paperwork, training, introductions, and culture immersion — to help them become productive and engaged quickly."},
        {"q": "Which law governs workplace discrimination in many countries?",
         "options": ["Contract Law", "Equal Employment Opportunity laws",
                     "Trade Union Act", "Corporate Governance Act"],
         "answer": "Equal Employment Opportunity laws",
         "explanation": "EEO (Equal Employment Opportunity) laws prohibit discrimination in hiring, firing, pay, or promotions based on race, gender, age, religion, disability, etc. Examples: Title VII (US), Equality Act (UK)."},
        {"q": "What is the purpose of a performance improvement plan (PIP)?",
         "options": ["Give promotions", "Track attendance",
                     "Help underperforming employees improve",
                     "Recruit new talent"],
         "answer": "Help underperforming employees improve",
         "explanation": "A PIP is a formal document outlining specific performance gaps, improvement goals, timelines, and support provided. It gives employees a clear path to meet expectations before more serious action is taken."},
        {"q": "What does 'talent acquisition' involve?",
         "options": ["Only posting job ads",
                     "Strategic process of finding, attracting and hiring talent",
                     "Training existing employees",
                     "Managing payroll"],
         "answer": "Strategic process of finding, attracting and hiring talent",
         "explanation": "Talent acquisition goes beyond recruitment — it includes employer branding, pipeline building, strategic workforce planning, and candidate experience to attract top talent aligned with long-term business goals."},
        {"q": "What is employee attrition?",
         "options": ["Hiring rate", "Rate at which employees leave an organisation",
                     "Training completion rate", "Promotion rate"],
         "answer": "Rate at which employees leave an organisation",
         "explanation": "Attrition rate = (employees who left / average headcount) × 100. High attrition is costly — replacing an employee can cost 50-200% of their annual salary. Understanding causes helps reduce it."},
        {"q": "Which interview method is considered most predictive of job performance?",
         "options": ["Unstructured interview", "Structured behavioral interview",
                     "Phone screening", "Group discussion"],
         "answer": "Structured behavioral interview",
         "explanation": "Structured behavioral interviews (using STAR method — Situation, Task, Action, Result) consistently show the highest predictive validity for job performance because they standardize questions and evaluation criteria."},
        {"q": "What is KPI in HR context?",
         "options": ["Key Personnel Information",
                     "Key Performance Indicator",
                     "Knowledge Process Integration",
                     "Key Payroll Index"],
         "answer": "Key Performance Indicator",
         "explanation": "KPIs are measurable values that show how effectively goals are being achieved. HR KPIs include time-to-hire, cost-per-hire, retention rate, employee engagement score, and training ROI."},
        {"q": "What does HRIS stand for?",
         "options": ["Human Resource Information System",
                     "HR Interview Scoring",
                     "Human Recruitment Intelligence Suite",
                     "HR Integration Service"],
         "answer": "Human Resource Information System",
         "explanation": "HRIS is software that centralises HR data — employee records, payroll, benefits, attendance, and performance. Examples: SAP SuccessFactors, Workday, BambooHR. It reduces manual work and improves data accuracy."},
        {"q": "What is a 360-degree feedback?",
         "options": ["Self-assessment only",
                     "Feedback from manager only",
                     "Feedback from peers, subordinates, and managers",
                     "Annual salary review"],
         "answer": "Feedback from peers, subordinates, and managers",
         "explanation": "360-degree feedback collects performance input from all directions — the employee themselves, their direct reports, peers, and managers — giving a comprehensive, multi-perspective view of their strengths and development areas."},
    ],
    "FINANCE": [
        {"q": "What does P&L stand for?",
         "options": ["Process & Logistics", "Profit & Loss", "Planning & Launch", "Price & Liability"],
         "answer": "Profit & Loss",
         "explanation": "The Profit & Loss (P&L) statement, also called the income statement, summarises revenues, costs, and expenses over a period to show net profit or loss. It's one of the three core financial statements."},
        {"q": "What is DCF analysis used for?",
         "options": ["Tracking expenses", "Valuing an investment based on future cash flows",
                     "Calculating payroll", "Measuring employee performance"],
         "answer": "Valuing an investment based on future cash flows",
         "explanation": "DCF (Discounted Cash Flow) estimates an investment's present value by discounting projected future cash flows at a rate reflecting risk and time value of money. If DCF > current price, the investment may be undervalued."},
        {"q": "What does GAAP stand for?",
         "options": ["Generally Accepted Accounting Principles",
                     "Government Asset Allocation Protocol",
                     "Gross Annual Audit Procedure",
                     "Global Accounting Audit Platform"],
         "answer": "Generally Accepted Accounting Principles",
         "explanation": "GAAP is a set of standardized accounting rules and procedures used in the US to ensure financial statements are consistent, comparable, and transparent. IFRS is the international equivalent."},
        {"q": "What is working capital?",
         "options": ["Long-term assets minus long-term liabilities",
                     "Current assets minus current liabilities",
                     "Total revenue minus total expenses",
                     "Net profit after tax"],
         "answer": "Current assets minus current liabilities",
         "explanation": "Working capital = Current Assets − Current Liabilities. It measures short-term liquidity — a company's ability to pay obligations due within one year. Positive working capital means the company can cover short-term debts."},
        {"q": "What is a balance sheet?",
         "options": ["Statement showing revenue and expenses",
                     "Snapshot of assets, liabilities and equity at a point in time",
                     "Cash flow projection",
                     "Payroll summary"],
         "answer": "Snapshot of assets, liabilities and equity at a point in time",
         "explanation": "The balance sheet follows the equation: Assets = Liabilities + Equity. It shows what a company owns (assets), what it owes (liabilities), and shareholders' stake (equity) at a specific date, unlike P&L which covers a period."},
    ],
    "SALES": [
        {"q": "What does CRM stand for?",
         "options": ["Customer Relationship Management", "Cost Reduction Method",
                     "Client Revenue Monitoring", "Corporate Resource Management"],
         "answer": "Customer Relationship Management",
         "explanation": "CRM systems (Salesforce, HubSpot) help sales teams track interactions with leads and customers, manage pipelines, automate follow-ups, and analyze data to improve conversion rates and retention."},
        {"q": "What is a sales funnel?",
         "options": ["A tool to filter leads",
                     "The stages a prospect goes through before becoming a customer",
                     "A pricing model",
                     "A commission structure"],
         "answer": "The stages a prospect goes through before becoming a customer",
         "explanation": "The sales funnel tracks prospects through stages: Awareness → Interest → Consideration → Intent → Purchase. Understanding drop-off at each stage helps sales teams improve conversion rates."},
        {"q": "What does B2B mean?",
         "options": ["Back to Business", "Business to Business",
                     "Budget to Billing", "Brand to Buyer"], "answer": "Business to Business",
         "explanation": "B2B (Business to Business) refers to companies selling to other businesses, not consumers. B2B sales typically involve longer cycles, multiple decision-makers, higher deal values, and relationship-driven selling."},
        {"q": "What is upselling?",
         "options": ["Selling to a new customer",
                     "Convincing a customer to buy a higher-end version",
                     "Discounting a product",
                     "Cold calling prospects"],
         "answer": "Convincing a customer to buy a higher-end version",
         "explanation": "Upselling encourages customers to buy a premium version or add-ons (e.g. upgrading from Standard to Pro). Cross-selling suggests related products. Both increase average order value and revenue per customer."},
        {"q": "What is a KPI example for sales?",
         "options": ["Number of emails sent",
                     "Monthly recurring revenue (MRR)",
                     "Office attendance",
                     "Number of meetings attended"],
         "answer": "Monthly recurring revenue (MRR)",
         "explanation": "MRR (Monthly Recurring Revenue) is a key SaaS/subscription sales metric measuring predictable monthly income. Other important sales KPIs include conversion rate, average deal size, pipeline value, and quota attainment."},
    ],
    "DEFAULT": [
        {"q": "What does SMART stand for in goal-setting?",
         "options": ["Specific, Measurable, Achievable, Relevant, Time-bound",
                     "Simple, Manageable, Accurate, Realistic, Timely",
                     "Strategic, Motivated, Actionable, Reasonable, Trackable",
                     "Structured, Monitored, Agile, Responsive, Tactical"],
         "answer": "Specific, Measurable, Achievable, Relevant, Time-bound",
         "explanation": "SMART goals ensure clarity and trackability. Specific = clear outcome. Measurable = quantifiable. Achievable = realistic. Relevant = aligned with objectives. Time-bound = has a deadline. This framework is used across all professions."},
        {"q": "What is KPI?",
         "options": ["Key Process Indicator", "Key Performance Indicator",
                     "Knowledge Protocol Index", "Key Personnel Information"],
         "answer": "Key Performance Indicator",
         "explanation": "KPIs are quantifiable metrics used to evaluate success against goals. Good KPIs are specific, measurable, and actionable. Examples: revenue growth, customer satisfaction score, project completion rate."},
        {"q": "What is the purpose of a cover letter?",
         "options": ["Replace your resume",
                     "Introduce yourself and explain why you're a good fit",
                     "List all your skills", "Provide references"],
         "answer": "Introduce yourself and explain why you're a good fit",
         "explanation": "A cover letter complements your resume by telling your story — why you want this specific role, how your experience addresses the employer's needs, and what you'll contribute. It shows communication skills and genuine interest."},
        {"q": "What does 'stakeholder' mean in a project?",
         "options": ["Only the project manager",
                     "Anyone with an interest or impact in the project",
                     "Only the client",
                     "Only the development team"],
         "answer": "Anyone with an interest or impact in the project",
         "explanation": "Stakeholders include everyone affected by or who can affect a project: sponsors, team members, clients, end users, regulators. Identifying and managing stakeholders is critical to project success."},
        {"q": "Which is a good practice in professional email communication?",
         "options": ["Use all caps for emphasis",
                     "Write clear subject lines and be concise",
                     "Always use emojis",
                     "CC everyone in the company"],
         "answer": "Write clear subject lines and be concise",
         "explanation": "Effective professional emails have a clear subject line summarising the purpose, concise body (readers scan, not read), clear action items, and appropriate tone. All caps, excessive emojis, and unnecessary CCs reduce professionalism."},
    ],
}

STATIC_INTERVIEW: dict[str, list[str]] = {
    "DATA-SCIENCE": [
        "Walk me through a data science project you've worked on end-to-end.",
        "How do you handle missing data in a dataset?",
        "Explain the difference between supervised and unsupervised learning.",
        "How would you explain a machine learning model's output to a non-technical stakeholder?",
        "Describe a time you found an unexpected insight in data — what did you do with it?",
        "What steps do you take to prevent overfitting in a model?",
        "How do you choose between different machine learning algorithms for a problem?",
        "Describe your experience with data visualization tools.",
        "How do you validate the quality of your data before building a model?",
        "Tell me about a time your analysis led to a business decision.",
    ],
    "WEB-DEVELOPER": [
        "Describe a challenging bug you fixed — how did you approach debugging it?",
        "How do you ensure your web applications are responsive and accessible?",
        "Walk me through your process of optimizing a slow web page.",
        "How do you stay updated with the latest web development trends?",
        "Describe a project where you had to learn a new framework or technology quickly.",
        "How do you approach writing maintainable, reusable code?",
        "Tell me about your experience with version control and team collaboration.",
        "How do you handle cross-browser compatibility issues?",
        "Describe your experience with RESTful API design or consumption.",
        "What's your approach to testing your code?",
    ],
    "HR": [
        "Describe a time you resolved a conflict between two employees.",
        "How do you ensure a fair and unbiased recruitment process?",
        "Tell me about a successful onboarding program you designed or implemented.",
        "How do you handle confidential employee information?",
        "Describe a situation where you had to deliver difficult feedback to an employee.",
        "How do you measure the effectiveness of your HR initiatives?",
        "Tell me about your experience with performance management systems.",
        "How do you stay updated with changes in employment law?",
        "Describe a time you had to manage a high volume of open positions simultaneously.",
        "How do you build a positive workplace culture?",
    ],
    "FINANCE": [
        "Walk me through how you would build a financial model from scratch.",
        "Describe a time you identified a cost-saving opportunity in a budget.",
        "How do you ensure accuracy when working with large financial datasets?",
        "Explain how you would present financial findings to a non-finance audience.",
        "Tell me about your experience with financial forecasting.",
        "Describe a situation where you had to work under a tight reporting deadline.",
        "How do you stay current with changes in accounting standards or regulations?",
        "Tell me about a time your financial analysis influenced a major business decision.",
        "How do you prioritize tasks during month-end close?",
        "Describe your experience with ERP systems like SAP or Oracle.",
    ],
    "SALES": [
        "Tell me about the largest deal you ever closed — how did you do it?",
        "How do you handle rejection from a prospect?",
        "Describe your process for qualifying a lead.",
        "How do you build long-term relationships with clients?",
        "Tell me about a time you exceeded your sales quota.",
        "How do you research a prospect before reaching out?",
        "Describe a situation where you turned a lost deal into a win.",
        "How do you prioritize your accounts and opportunities?",
        "What strategies do you use for upselling or cross-selling?",
        "Tell me about a challenging client you managed successfully.",
    ],
    "DEFAULT": [
        "Tell me about yourself and your professional background.",
        "What are your greatest strengths and how have they helped you in your career?",
        "Describe a challenging situation at work and how you resolved it.",
        "Where do you see yourself professionally in 5 years?",
        "Tell me about a time you worked effectively in a team.",
        "How do you prioritize tasks when you have multiple deadlines?",
        "Describe a time you showed initiative beyond your job description.",
        "How do you handle feedback and criticism?",
        "Tell me about a project you're particularly proud of.",
        "Why are you interested in this role?",
    ],
}


# ── FLAN-T5 loader (lazy — only loaded on first call) ─────────────────────────
_flan_pipeline = None
_flan_load_attempted = False


def _load_flan() -> Optional[object]:
    """Lazy-load FLAN-T5 small. Returns pipeline or None on failure."""
    global _flan_pipeline, _flan_load_attempted
    if _flan_load_attempted:
        return _flan_pipeline
    _flan_load_attempted = True
    try:
        from transformers import pipeline  # type: ignore[import]
        print("[question_generator] Loading FLAN-T5-small from HuggingFace cache...")
        _flan_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_new_tokens=200,
        )
        print("[question_generator] FLAN-T5 loaded successfully.")
    except Exception as e:
        print(f"[question_generator] FLAN-T5 not available: {e}")
        _flan_pipeline = None
    return _flan_pipeline


# ── Groq fallback (LLaMA-3 free tier) ─────────────────────────────────────────

def _groq_generate(prompt: str) -> Optional[str]:
    """Call Groq API free tier (LLaMA-3). Returns text or None."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq  # type: ignore[import]
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[question_generator] Groq error: {e}")
        return None


# ── FLAN-T5 question generation ────────────────────────────────────────────────

def _flan_generate_mcq(role: str, skill: str) -> Optional[dict]:
    """Ask FLAN-T5 to generate a single MCQ for a given skill."""
    pipe = _load_flan()
    if not pipe:
        return None
    prompt = (
        f"Generate a multiple choice question for a {role} job interview about {skill}. "
        f"Format: Question: ... Options: A) ... B) ... C) ... D) ... Answer: ..."
    )
    try:
        result = pipe(prompt, max_new_tokens=150)[0]["generated_text"]
        # Parse question/options/answer from result
        q_match = re.search(r'Question:\s*(.+?)(?:Options:|$)', result, re.IGNORECASE | re.DOTALL)
        opts_match = re.findall(r'[A-D]\)\s*(.+?)(?=[A-D]\)|Answer:|$)', result, re.DOTALL)
        ans_match = re.search(r'Answer:\s*([A-D]\)?\s*.+?)$', result, re.IGNORECASE)
        if q_match and len(opts_match) >= 3:
            question = q_match.group(1).strip()
            options = [o.strip() for o in opts_match[:4]]
            answer = ans_match.group(1).strip() if ans_match else options[0]
            # Clean answer to match option text
            for opt in options:
                if opt.lower() in answer.lower() or answer.lower() in opt.lower():
                    answer = opt
                    break
            return {"q": question, "options": options, "answer": answer, "source": "flan-t5"}
    except Exception as e:
        print(f"[question_generator] FLAN-T5 parse error: {e}")
    return None


def _flan_generate_interview_q(role: str, skill: str) -> Optional[str]:
    """Ask FLAN-T5 to generate a behavioral interview question."""
    pipe = _load_flan()
    if not pipe:
        return None
    prompt = (
        f"Generate one behavioral interview question for a {role} professional "
        f"that tests their {skill} skill. Start with 'Tell me about' or 'Describe a time'."
    )
    try:
        result = pipe(prompt, max_new_tokens=80)[0]["generated_text"].strip()
        if len(result) > 20:
            return result
    except Exception:
        pass
    return None


# ── Groq-based generation ──────────────────────────────────────────────────────

def _groq_generate_questions(role: str, skills: list, mcq_count: int, interview_count: int) -> Optional[dict]:
    """Generate questions via Groq/LLaMA-3 as JSON."""
    skills_str = ", ".join(skills[:6]) if skills else role
    prompt = f"""Generate interview questions for a {role} role.
Skills to focus on: {skills_str}

Return ONLY valid JSON in this exact format:
{{
  "mcq": [
    {{"q": "question text", "options": ["A", "B", "C", "D"], "answer": "correct option text"}}
  ],
  "interview": ["question 1", "question 2"]
}}

Generate {min(mcq_count, 5)} MCQ questions and {min(interview_count, 5)} interview questions."""

    raw = _groq_generate(prompt)
    if not raw:
        return None
    try:
        # Extract JSON block
        json_match = re.search(r'\{[\s\S]+\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data
    except Exception as e:
        print(f"[question_generator] Groq JSON parse error: {e}")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_static_mcq(role: str, count: int = 10) -> list[dict]:
    """Return shuffled static MCQ questions for the given role."""
    bank = STATIC_MCQ.get(role, STATIC_MCQ["DEFAULT"])
    questions = bank.copy()
    random.shuffle(questions)
    # Shuffle options within each question (keep answer tracking correct)
    result = []
    for item in questions[:count]:
        opts = item["options"].copy()
        random.shuffle(opts)
        result.append({
            "q":       item["q"],
            "options": opts,
            "answer":  item["answer"],
            "source":  "static",
        })
    return result


def get_interview_questions(role: str, experience_years: int = 2,
                            skills: list = None, count: int = 10) -> list[str]:
    """Return interview questions for the given role."""
    bank = STATIC_INTERVIEW.get(role, STATIC_INTERVIEW["DEFAULT"])
    questions = bank.copy()
    random.shuffle(questions)
    return questions[:count]


def generate_questions_for_resume(
    role: str,
    skills: list = None,
    experience_years: int = 2,
    mcq_count: int = 10,
    interview_count: int = 10,
) -> dict:
    """
    Main entry point. Tries FLAN-T5 → Groq → static fallback.
    Returns: { mcq, interview, role, model_used, skills_used }
    """
    skills = skills or []
    model_used = "static"
    mcq_questions = []
    interview_questions_list = []

    # ── Try Groq first (better quality, still free) ───────────────────────
    if os.environ.get("GROQ_API_KEY"):
        groq_result = _groq_generate_questions(role, skills, mcq_count, interview_count)
        if groq_result:
            mcq_questions        = groq_result.get("mcq", [])[:mcq_count]
            interview_questions_list = groq_result.get("interview", [])[:interview_count]
            model_used = "groq-llama3"

    # ── Try FLAN-T5 for any remaining questions ───────────────────────────
    if len(mcq_questions) < mcq_count or len(interview_questions_list) < interview_count:
        pipe = _load_flan()
        if pipe and skills:
            skill_sample = random.sample(skills, min(len(skills), 4))
            for skill in skill_sample:
                if len(mcq_questions) < mcq_count:
                    q = _flan_generate_mcq(role, skill)
                    if q:
                        mcq_questions.append(q)
                if len(interview_questions_list) < interview_count:
                    q = _flan_generate_interview_q(role, skill)
                    if q:
                        interview_questions_list.append(q)
            if mcq_questions or interview_questions_list:
                model_used = "flan-t5"

    # ── Fill remaining slots from static bank ─────────────────────────────
    static_mcq  = get_static_mcq(role, mcq_count)
    static_ivw  = get_interview_questions(role, experience_years, skills, interview_count)

    # Pad with static if AI generation didn't fill quota
    if len(mcq_questions) < mcq_count:
        needed = mcq_count - len(mcq_questions)
        # Only add static questions not already present
        existing_q_texts = {q.get("q", "").lower() for q in mcq_questions}
        for sq in static_mcq:
            if sq["q"].lower() not in existing_q_texts and needed > 0:
                mcq_questions.append(sq)
                needed -= 1

    if len(interview_questions_list) < interview_count:
        needed = interview_count - len(interview_questions_list)
        existing_ivw = {q.lower() for q in interview_questions_list}
        for sq in static_ivw:
            if sq.lower() not in existing_ivw and needed > 0:
                interview_questions_list.append(sq)
                needed -= 1

    return {
        "mcq":          mcq_questions[:mcq_count],
        "interview":    interview_questions_list[:interview_count],
        "role":         role,
        "model_used":   model_used,
        "skills_used":  skills[:6],
    } 