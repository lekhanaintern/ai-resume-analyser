"""
nlp_engine.py — Advanced NLP Resume Enhancement Engine
=======================================================
Zero external-LLM dependencies. Uses:
  - Custom POS-style verb/phrase detection (regex + curated lexicons)
  - TF-IDF cosine similarity (sklearn) for semantic skill matching
  - Dependency-style sentence restructuring for bullet rewrites
  - Role taxonomy with gap scoring
  - ATS compliance engine

Drop-in module. Call: enhance_resume_for_role(resume_text, target_role)
Returns a rich dict with enhanced resume, skill map, section scores, diff stats.
"""

import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. ROLE TAXONOMY
#    Each role: required_skills (must-have), preferred_skills (nice-to-have),
#    action_verbs (domain verbs), summary_traits, achievement_patterns
# ─────────────────────────────────────────────────────────────
ROLE_TAXONOMY = {
    "DATA-SCIENCE": {
        "title": "Data Scientist",
        "required_skills": [
            "python","machine learning","sql","data analysis","statistics",
            "scikit-learn","pandas","numpy","data visualization","model deployment"
        ],
        "preferred_skills": [
            "tensorflow","pytorch","deep learning","nlp","spark","hadoop",
            "tableau","power bi","r","feature engineering","a/b testing",
            "jupyter","mlflow","aws","azure","gcp"
        ],
        "action_verbs": [
            "analyzed","modeled","predicted","classified","clustered","optimized",
            "automated","visualized","extracted","engineered","trained","deployed",
            "evaluated","validated","benchmarked","transformed","processed"
        ],
        "domain_nouns": [
            "dataset","model","pipeline","algorithm","feature","accuracy","precision",
            "recall","f1","roc","auc","regression","classification","clustering",
            "neural network","insight","prediction","inference","metric"
        ],
        "summary_traits": [
            "data-driven","analytical","detail-oriented","statistically rigorous",
            "business-impact focused","model-first thinker"
        ],
        "achievement_patterns": [
            "improved model accuracy by {n}%",
            "reduced processing time by {n}%",
            "built pipeline processing {n}M+ records",
            "generated ${n}K in cost savings through predictive analytics",
            "deployed {n}+ ML models into production"
        ],
        "ats_keywords": [
            "machine learning","data science","python","sql","statistical modeling",
            "predictive analytics","data pipeline","model training","feature engineering"
        ]
    },
    "WEB-DEVELOPER": {
        "title": "Web Developer",
        "required_skills": [
            "html5","css3","javascript","responsive design","git","rest api","react"
        ],
        "preferred_skills": [
            "typescript","node.js","mongodb","postgresql","docker","webpack","redux",
            "graphql","next.js","vue","angular","express","aws","ci/cd","jest"
        ],
        "action_verbs": [
            "developed","built","designed","implemented","integrated","optimized",
            "deployed","refactored","tested","debugged","shipped","architected",
            "migrated","automated","maintained","documented"
        ],
        "domain_nouns": [
            "web application","api","endpoint","component","frontend","backend",
            "database","performance","load time","uptime","ui","ux","interface",
            "repository","deployment","sprint","release"
        ],
        "summary_traits": [
            "full-stack capable","performance-obsessed","clean-code advocate",
            "user-experience focused","agile practitioner"
        ],
        "achievement_patterns": [
            "reduced page load time by {n}%",
            "built {n}+ web applications serving {n}K users",
            "improved uptime to {n}.{n}%",
            "cut deployment time by {n}% through CI/CD automation",
            "increased test coverage to {n}%"
        ],
        "ats_keywords": [
            "web development","javascript","react","html","css","frontend",
            "backend","full stack","rest api","responsive design","git"
        ]
    },
    "SOFTWARE-ENGINEER": {
        "title": "Software Engineer",
        "required_skills": [
            "python","java","c++","data structures","algorithms","git",
            "system design","sql","oop","agile"
        ],
        "preferred_skills": [
            "docker","kubernetes","microservices","aws","ci/cd","linux",
            "rest api","kafka","redis","elasticsearch","spring","django"
        ],
        "action_verbs": [
            "architected","engineered","designed","implemented","optimized",
            "refactored","automated","deployed","scaled","shipped","led",
            "reviewed","mentored","resolved","integrated","maintained"
        ],
        "domain_nouns": [
            "microservice","architecture","system","codebase","api","latency",
            "throughput","uptime","test coverage","sprint","release","module",
            "repository","deployment","incident","bug","feature"
        ],
        "summary_traits": [
            "systems thinker","scalability-focused","test-driven",
            "clean architecture advocate","collaborative engineer"
        ],
        "achievement_patterns": [
            "reduced system latency by {n}%",
            "scaled service to handle {n}M+ daily requests",
            "decreased bug rate by {n}% through rigorous testing",
            "led team of {n} engineers to deliver project on time",
            "reduced deployment time from {n} hours to {n} minutes"
        ],
        "ats_keywords": [
            "software engineering","python","java","system design","algorithms",
            "microservices","agile","docker","kubernetes","ci/cd","rest api"
        ]
    },
    "HR": {
        "title": "HR Professional",
        "required_skills": [
            "talent acquisition","recruitment","onboarding","hris",
            "performance management","employee relations","payroll"
        ],
        "preferred_skills": [
            "hr analytics","compensation","benefits administration","training",
            "conflict resolution","ats","labor law","succession planning",
            "employer branding","diversity equity inclusion"
        ],
        "action_verbs": [
            "recruited","onboarded","managed","developed","implemented","facilitated",
            "coordinated","resolved","designed","administered","trained","evaluated",
            "negotiated","streamlined","improved","reduced"
        ],
        "domain_nouns": [
            "candidate","employee","workforce","retention","engagement","culture",
            "policy","procedure","headcount","attrition","time-to-hire","satisfaction"
        ],
        "summary_traits": [
            "people-first mindset","relationship builder","compliance-aware",
            "empathetic leader","data-informed HR professional"
        ],
        "achievement_patterns": [
            "reduced time-to-hire by {n}%",
            "improved employee retention by {n}%",
            "onboarded {n}+ employees per quarter",
            "achieved {n}% employee satisfaction score",
            "managed HR operations for {n}+ headcount"
        ],
        "ats_keywords": [
            "human resources","talent acquisition","recruitment","onboarding",
            "employee relations","hris","performance management","payroll"
        ]
    },
    "DESIGNER": {
        "title": "UI/UX Designer",
        "required_skills": [
            "figma","ui/ux design","wireframing","prototyping","user research",
            "typography","design systems","accessibility"
        ],
        "preferred_skills": [
            "adobe xd","photoshop","illustrator","invision","sketch","motion design",
            "usability testing","information architecture","color theory","html","css"
        ],
        "action_verbs": [
            "designed","prototyped","researched","wireframed","iterated","tested",
            "presented","collaborated","delivered","crafted","optimized","created",
            "redesigned","implemented","documented","facilitated"
        ],
        "domain_nouns": [
            "wireframe","prototype","mockup","user flow","persona","journey map",
            "design system","component","style guide","usability","conversion",
            "drop-off","engagement","feedback","sprint","handoff"
        ],
        "summary_traits": [
            "user-centric","visually driven","research-grounded",
            "pixel-perfect","empathy-led designer"
        ],
        "achievement_patterns": [
            "improved user satisfaction score by {n}%",
            "reduced onboarding drop-off by {n}%",
            "delivered designs for {n}+ products",
            "conducted {n}+ user research sessions",
            "increased conversion rate by {n}%"
        ],
        "ats_keywords": [
            "ui design","ux design","figma","wireframing","prototyping",
            "user research","design systems","accessibility","usability"
        ]
    },
    "FINANCE": {
        "title": "Finance Professional",
        "required_skills": [
            "financial analysis","excel","budgeting","forecasting","gaap",
            "variance analysis","financial modeling","p&l"
        ],
        "preferred_skills": [
            "dcf","tableau","power bi","sap","erp","risk assessment",
            "compliance","ifrs","balance sheet","cash flow","investment analysis"
        ],
        "action_verbs": [
            "analyzed","forecasted","modeled","prepared","managed","optimized",
            "reported","reconciled","identified","reduced","increased","audited",
            "evaluated","streamlined","presented","allocated"
        ],
        "domain_nouns": [
            "budget","forecast","revenue","cost","variance","margin","ebitda",
            "cash flow","balance sheet","p&l","roi","kpi","headcount","capex","opex"
        ],
        "summary_traits": [
            "financially rigorous","detail-oriented","strategic thinker",
            "compliance-aware","data-driven finance professional"
        ],
        "achievement_patterns": [
            "managed ${n}M+ annual budget",
            "identified ${n}K in cost savings",
            "reduced month-end close by {n} days",
            "improved forecast accuracy by {n}%",
            "prepared financial reports for {n}+ stakeholders"
        ],
        "ats_keywords": [
            "financial analysis","budgeting","forecasting","excel","gaap",
            "financial modeling","p&l","variance analysis","reporting"
        ]
    },
    "INFORMATION-TECHNOLOGY": {
        "title": "IT Professional",
        "required_skills": [
            "network administration","troubleshooting","windows server","linux",
            "active directory","helpdesk","tcp/ip","cybersecurity","cloud computing"
        ],
        "preferred_skills": [
            "aws","azure","vmware","itil","firewall","vpn","backup recovery",
            "siem","endpoint security","patch management","scripting","powershell"
        ],
        "action_verbs": [
            "administered","configured","deployed","maintained","troubleshot","secured",
            "monitored","upgraded","migrated","automated","resolved","implemented",
            "managed","optimized","documented","trained"
        ],
        "domain_nouns": [
            "infrastructure","endpoint","server","network","uptime","incident",
            "ticket","sla","patch","vulnerability","backup","recovery","latency"
        ],
        "summary_traits": [
            "infrastructure-focused","security-conscious","problem-solver",
            "sla-committed","reliability-driven IT professional"
        ],
        "achievement_patterns": [
            "maintained {n}.{n}% system uptime",
            "managed IT infrastructure for {n}+ endpoints",
            "reduced incident resolution time by {n}%",
            "cut infrastructure costs by {n}% through cloud migration",
            "resolved {n}+ tickets per week with {n}% SLA compliance"
        ],
        "ats_keywords": [
            "information technology","network administration","troubleshooting",
            "cybersecurity","cloud computing","active directory","linux","itil"
        ]
    },
    "SALES": {
        "title": "Sales Professional",
        "required_skills": [
            "lead generation","crm","pipeline management","negotiation",
            "account management","b2b","client retention","salesforce"
        ],
        "preferred_skills": [
            "b2c","cold outreach","upselling","cross-selling","quota achievement",
            "revenue growth","forecasting","hubspot","linkedin sales navigator"
        ],
        "action_verbs": [
            "exceeded","generated","closed","grew","built","managed","negotiated",
            "converted","retained","prospected","presented","identified",
            "accelerated","delivered","achieved","expanded"
        ],
        "domain_nouns": [
            "revenue","quota","pipeline","prospect","deal","account","client",
            "territory","target","conversion","churn","upsell","forecast","roi"
        ],
        "summary_traits": [
            "target-driven","relationship builder","hunter-farmer balance",
            "revenue-focused","consultative seller"
        ],
        "achievement_patterns": [
            "exceeded quarterly quota by {n}%",
            "generated ${n}M+ in new revenue",
            "converted {n}% of prospects to closed deals",
            "retained {n}% of key accounts year-over-year",
            "built pipeline of {n}+ qualified prospects"
        ],
        "ats_keywords": [
            "sales","lead generation","crm","pipeline","account management",
            "revenue","negotiation","b2b","salesforce","quota"
        ]
    },
    "HEALTHCARE": {
        "title": "Healthcare Professional",
        "required_skills": [
            "patient care","clinical procedures","emr","ehr","hipaa compliance",
            "medical records","vital signs","cpr/bls","care coordination"
        ],
        "preferred_skills": [
            "triage","icd-10","medical billing","medication administration",
            "patient education","infection control","wound care","iv therapy"
        ],
        "action_verbs": [
            "administered","assessed","coordinated","delivered","documented",
            "educated","facilitated","implemented","maintained","managed",
            "monitored","provided","supported","treated","trained","reduced"
        ],
        "domain_nouns": [
            "patient","caseload","care plan","diagnosis","treatment","compliance",
            "documentation","protocol","outcome","readmission","satisfaction"
        ],
        "summary_traits": [
            "patient-centered","compassionate","clinically precise",
            "compliance-driven","evidence-based practitioner"
        ],
        "achievement_patterns": [
            "managed caseload of {n}+ patients daily",
            "reduced patient wait time by {n}%",
            "achieved {n}% HIPAA compliance rate",
            "decreased readmission rate by {n}%",
            "maintained zero critical error record for {n} months"
        ],
        "ats_keywords": [
            "patient care","clinical","emr","hipaa","healthcare","medical",
            "care coordination","documentation","compliance","cpr"
        ]
    },
    "ENGINEERING": {
        "title": "Engineer",
        "required_skills": [
            "cad","project management","quality control","technical documentation",
            "root cause analysis","process improvement","iso standards"
        ],
        "preferred_skills": [
            "autocad","solidworks","six sigma","lean","fmea","manufacturing",
            "matlab","plc","3d modeling","tolerance analysis","project scheduling"
        ],
        "action_verbs": [
            "designed","engineered","developed","tested","optimized","implemented",
            "analyzed","led","improved","managed","coordinated","delivered",
            "reduced","increased","documented","validated"
        ],
        "domain_nouns": [
            "design","prototype","specification","tolerance","defect","yield",
            "cycle time","throughput","compliance","scope","milestone","budget"
        ],
        "summary_traits": [
            "precision-driven","quality-focused","analytical problem-solver",
            "delivery-oriented","process improvement mindset"
        ],
        "achievement_patterns": [
            "improved production yield by {n}%",
            "reduced defect rate by {n}%",
            "delivered {n}+ projects within scope and budget",
            "cut cycle time by {n}% through process redesign",
            "achieved ISO {n}001 compliance across all deliverables"
        ],
        "ats_keywords": [
            "engineering","cad","quality control","project management",
            "process improvement","technical documentation","iso","root cause analysis"
        ]
    },
    "ACCOUNTANT": {
        "title": "Accountant",
        "required_skills": [
            "gaap","tax preparation","auditing","financial reporting",
            "reconciliation","accounts payable","accounts receivable","excel"
        ],
        "preferred_skills": [
            "quickbooks","sap","tally","gst","tds","cost accounting","erp",
            "payroll","budgeting","internal controls","ifrs"
        ],
        "action_verbs": [
            "prepared","reconciled","audited","managed","reported","processed",
            "analyzed","reviewed","filed","reduced","improved","maintained",
            "identified","recovered","streamlined","implemented"
        ],
        "domain_nouns": [
            "balance sheet","ledger","trial balance","tax return","audit","invoice",
            "payable","receivable","journal","variance","compliance","close cycle"
        ],
        "summary_traits": [
            "meticulous","compliance-focused","numbers-driven",
            "deadline-oriented","detail-obsessed accountant"
        ],
        "achievement_patterns": [
            "managed accounts for {n}+ clients with 100% accuracy",
            "reduced month-end close from {n} to {n} days",
            "recovered ${n}K in billing discrepancies",
            "filed {n}+ tax returns with zero penalties",
            "improved audit compliance score to {n}%"
        ],
        "ats_keywords": [
            "accounting","gaap","tax","auditing","financial reporting",
            "reconciliation","accounts payable","accounts receivable","quickbooks"
        ]
    },
    "BANKING": {
        "title": "Banking Professional",
        "required_skills": [
            "kyc","aml","risk management","financial products","regulatory compliance",
            "credit analysis","customer due diligence","loan processing"
        ],
        "preferred_skills": [
            "swift","trade finance","basel iii","treasury","wealth management",
            "investment banking","portfolio management","financial modeling"
        ],
        "action_verbs": [
            "processed","analyzed","managed","evaluated","ensured","maintained",
            "grew","cross-sold","retained","identified","resolved","reported",
            "onboarded","assessed","implemented","monitored"
        ],
        "domain_nouns": [
            "loan","portfolio","client","compliance","risk","audit","regulation",
            "kyc","aml","transaction","account","credit","collateral","npa"
        ],
        "summary_traits": [
            "compliance-first","relationship-driven","risk-aware",
            "client-centric","financially astute banking professional"
        ],
        "achievement_patterns": [
            "maintained {n}% regulatory compliance across {n}+ accounts",
            "grew client portfolio by {n}%",
            "processed {n}+ loan applications monthly",
            "reduced NPA rate by {n}%",
            "cross-sold {n}+ financial products per quarter"
        ],
        "ats_keywords": [
            "banking","kyc","aml","risk management","compliance","credit analysis",
            "financial products","regulatory","loan processing","portfolio"
        ]
    },
    "BUSINESS-DEVELOPMENT": {
        "title": "Business Development Professional",
        "required_skills": [
            "market research","lead generation","partnership development",
            "revenue growth","client acquisition","crm","negotiation","strategic planning"
        ],
        "preferred_skills": [
            "proposal writing","kpi tracking","salesforce","market analysis",
            "competitor analysis","go-to-market","pricing strategy","forecasting"
        ],
        "action_verbs": [
            "identified","built","closed","grew","developed","negotiated","launched",
            "expanded","generated","secured","established","delivered","accelerated",
            "researched","presented","managed"
        ],
        "domain_nouns": [
            "partnership","revenue","deal","pipeline","market","client","opportunity",
            "proposal","contract","territory","target","growth","roi","kpi"
        ],
        "summary_traits": [
            "growth-oriented","strategic relationship builder","revenue-obsessed",
            "hunter mentality","market development expert"
        ],
        "achievement_patterns": [
            "closed ${n}M+ in new partnerships",
            "grew revenue by {n}% year-over-year",
            "built partner network of {n}+ organizations",
            "identified {n}+ market expansion opportunities",
            "reduced deal cycle time by {n}%"
        ],
        "ats_keywords": [
            "business development","lead generation","revenue growth","partnerships",
            "market research","negotiation","crm","strategic planning","client acquisition"
        ]
    },
    "DIGITAL-MEDIA": {
        "title": "Digital Media Professional",
        "required_skills": [
            "content strategy","seo","social media marketing","google analytics",
            "copywriting","brand management","content creation","campaign management"
        ],
        "preferred_skills": [
            "sem","email marketing","adobe creative suite","wordpress","hubspot",
            "facebook ads","instagram","tiktok","influencer marketing","a/b testing"
        ],
        "action_verbs": [
            "created","grew","managed","launched","optimized","produced","drove",
            "increased","developed","executed","analyzed","collaborated","designed",
            "published","tracked","delivered"
        ],
        "domain_nouns": [
            "content","campaign","engagement","reach","impressions","roas","ctr",
            "follower","seo","keyword","brand","audience","conversion","funnel"
        ],
        "summary_traits": [
            "creative storyteller","data-informed content strategist",
            "brand-obsessed","audience-first","performance marketing mindset"
        ],
        "achievement_patterns": [
            "grew social following by {n}%",
            "achieved {n}x ROAS on paid campaigns",
            "increased organic traffic by {n}%",
            "produced {n}+ high-performing content pieces per month",
            "managed ${n}K+ ad spend with measurable ROI"
        ],
        "ats_keywords": [
            "digital marketing","content strategy","seo","social media","google analytics",
            "copywriting","brand management","campaign management","email marketing"
        ]
    },
    "DEFAULT": {
        "title": "Professional",
        "required_skills": [
            "communication","project management","problem solving","teamwork",
            "time management","microsoft office","reporting","presentation"
        ],
        "preferred_skills": [
            "leadership","negotiation","strategic planning","data analysis","excel",
            "stakeholder management","process improvement","documentation"
        ],
        "action_verbs": [
            "managed","led","developed","implemented","coordinated","delivered",
            "improved","analyzed","created","executed","built","optimized"
        ],
        "domain_nouns": [
            "project","team","stakeholder","deadline","objective","target","outcome",
            "process","report","presentation","milestone","budget"
        ],
        "summary_traits": [
            "results-driven","collaborative","adaptable","detail-oriented","proactive"
        ],
        "achievement_patterns": [
            "delivered {n}+ projects on time and within budget",
            "improved team efficiency by {n}%",
            "managed cross-functional team of {n}+ members",
            "reduced operational costs by {n}%"
        ],
        "ats_keywords": [
            "project management","communication","leadership","teamwork",
            "problem solving","microsoft office","reporting","analysis"
        ]
    }
}


# ─────────────────────────────────────────────────────────────
# 2. NLP UTILITIES
# ─────────────────────────────────────────────────────────────

# Weak verb patterns → strong replacements (regex → replacement)
WEAK_VERB_PATTERNS = [
    (r'\bresponsible for\b',       'Led'),
    (r'\bhelped (?:to |with )?',   'Supported '),
    (r'\bwas involved in\b',       'Contributed to'),
    (r'\bworked on\b',             'Developed'),
    (r'\bworked with\b',           'Collaborated with'),
    (r'\bworked under\b',          'Operated under'),
    (r'\bdid\b',                   'Executed'),
    (r'\bmade\b',                  'Created'),
    (r'\bhandled\b',               'Managed'),
    (r'\bassisted (?:in |with )?', 'Supported '),
    (r'\bparticipated in\b',       'Contributed to'),
    (r'\bwas part of\b',           'Contributed to'),
    (r'\btried to\b',              'Worked to'),
    (r'\battempted to\b',          'Worked to'),
    (r'\bwas responsible for\b',   'Oversaw'),
    (r'\bused to\b',               'Utilized'),
    (r'\bhad to\b',                'Was required to'),
    (r'\bgot to\b',                'Was selected to'),
    (r'\bwas tasked with\b',       'Led'),
    (r'\bsupported the\b',         'Contributed to the'),
]

# Passive voice detector
PASSIVE_PATTERNS = [
    r'\bwas (?:being )?(?:developed|built|created|managed|led|designed|implemented)\b',
    r'\bwere (?:being )?(?:developed|built|created|managed|led|designed|implemented)\b',
    r'\bhas been (?:developed|built|created|managed|led|designed|implemented)\b',
]

# Filler phrases that weaken bullets
FILLER_PATTERNS = [
    r'\bin order to\b', r'\bdue to the fact that\b', r'\bat this point in time\b',
    r'\bin the event that\b', r'\bfor the purpose of\b', r'\bwith regard to\b',
    r'\bin terms of\b', r'\ba number of\b', r'\bvarious\b', r'\bseveral\b',
    r'\bdiverse\b', r'\bextensive\b(?! experience)',
]

# Number patterns to detect quantification
NUMBER_PATTERNS = [
    r'\b\d+\s*%',                    # percentages
    r'\$\s*\d+[\d,.kKmMbB]*',       # money
    r'\b\d+[\d,]*\s*(?:users?|clients?|customers?|employees?|members?|projects?|products?|applications?|endpoints?|tickets?|records?|requests?|transactions?)\b',
    r'\b\d+x\b',                     # multipliers
    r'\b\d+\s*(?:days?|weeks?|months?|hours?)\b',  # time
]


def _count_metrics(text: str) -> int:
    """Count quantified achievements in a text."""
    count = 0
    for pattern in NUMBER_PATTERNS:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def _has_passive_voice(line: str) -> bool:
    for p in PASSIVE_PATTERNS:
        if re.search(p, line, re.IGNORECASE):
            return True
    return False


def _starts_with_weak_verb(line: str) -> bool:
    weak_starts = [
        r'^responsible for', r'^helped', r'^assisted', r'^participated',
        r'^was involved', r'^was part of', r'^worked on', r'^worked with',
        r'^did ', r'^made ', r'^handled ', r'^tried to', r'^attempted'
    ]
    for p in weak_starts:
        if re.match(p, line.strip(), re.IGNORECASE):
            return True
    return False


def _starts_with_strong_verb(line: str, role_verbs: list) -> bool:
    first_word = line.strip().split()[0].lower().rstrip('edy') if line.strip().split() else ''
    strong_verbs = set([
        'led','managed','built','designed','developed','created','implemented',
        'architected','delivered','launched','grew','generated','reduced',
        'increased','improved','optimized','automated','deployed','shipped',
        'coordinated','facilitated','negotiated','secured','established',
        'executed','achieved','exceeded','analyzed','engineered','scaled',
        'trained','mentored','streamlined','transformed','accelerated'
    ] + [v.rstrip('ed') for v in role_verbs])
    return first_word in strong_verbs or line.strip().split()[0].lower() in strong_verbs


def _inject_metric_if_missing(sentence: str, role_key: str) -> str:
    """
    If a bullet has no metric, intelligently inject a plausible one
    based on the sentence's domain context.
    """
    if _count_metrics(sentence) > 0:
        return sentence

    s = sentence.lower()
    # Choose metric type based on context
    if any(w in s for w in ['time','duration','speed','fast','slow','hours','days']):
        n = random.choice([15, 20, 25, 30, 35, 40])
        return sentence.rstrip('.') + f', reducing time by {n}%.'
    elif any(w in s for w in ['cost','budget','expense','saving','spend']):
        n = random.choice([10, 15, 20, 25])
        return sentence.rstrip('.') + f', cutting costs by {n}%.'
    elif any(w in s for w in ['user','customer','client','team','member','employee','stakeholder']):
        n = random.choice([5, 8, 10, 12, 15, 20, 25])
        return sentence.rstrip('.') + f', impacting {n}+ stakeholders.'
    elif any(w in s for w in ['error','bug','defect','issue','problem','incident','fail']):
        n = random.choice([20, 25, 30, 35, 40])
        return sentence.rstrip('.') + f', reducing error rate by {n}%.'
    elif any(w in s for w in ['process','workflow','pipeline','system','platform','tool']):
        n = random.choice([20, 25, 30, 35])
        return sentence.rstrip('.') + f', improving efficiency by {n}%.'
    elif any(w in s for w in ['revenue','sales','growth','profit','conversion']):
        n = random.choice([10, 15, 20, 25, 30])
        return sentence.rstrip('.') + f', contributing to {n}% revenue growth.'
    else:
        n = random.choice([15, 20, 25, 30])
        return sentence.rstrip('.') + f', achieving a {n}% improvement in outcomes.'


def _rewrite_bullet(line: str, role_verbs: list, role_key: str) -> tuple:
    """
    Rewrite a single bullet point.
    Returns (rewritten_line, list_of_changes_made)
    """
    changes = []
    original = line.strip()
    result = original

    # Strip leading bullet markers
    result = re.sub(r'^[-•●▪◦◆\*]\s*', '', result).strip()

    # 1. Fix weak verb openings
    for pattern, replacement in WEAK_VERB_PATTERNS:
        new = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if new != result:
            changes.append('Weak verb replaced')
            result = new
            break

    # 2. Remove filler phrases
    for fp in FILLER_PATTERNS:
        new = re.sub(fp, '', result, flags=re.IGNORECASE)
        if new != result:
            result = re.sub(r'  +', ' ', new).strip()
            changes.append('Filler removed')

    # 3. Fix passive voice (simple cases)
    if _has_passive_voice(result):
        result = re.sub(r'\bwas (developed|built|created|managed)\b', lambda m: m.group(1).capitalize() + 'd', result, flags=re.IGNORECASE)
        changes.append('Passive voice corrected')

    # 4. Ensure starts with a strong past-tense verb if applicable
    words = result.split()
    if words and not _starts_with_strong_verb(result, role_verbs):
        # Check if it looks like a job responsibility line
        if len(words) > 4 and any(c.islower() for c in words[0]):
            verb = random.choice(role_verbs[:8])
            # Capitalize and prepend
            result = verb.capitalize() + ' ' + result[0].lower() + result[1:]
            changes.append('Strong verb added')

    # 5. Inject metric if none present
    new = _inject_metric_if_missing(result, role_key)
    if new != result:
        result = new
        changes.append('Metric injected')

    # 6. Ensure proper capitalization and ending period
    if result:
        result = result[0].upper() + result[1:]
    if result and not result.endswith('.'):
        result += '.'

    return result, changes


# ─────────────────────────────────────────────────────────────
# 3. SECTION PARSER
# ─────────────────────────────────────────────────────────────
SECTION_HEADER_RE = re.compile(
    r'^(PROFESSIONAL SUMMARY|SUMMARY|OBJECTIVE|CAREER OBJECTIVE|PROFILE|ABOUT ME|'
    r'SKILLS|TECHNICAL SKILLS|KEY SKILLS|CORE COMPETENCIES|COMPETENCIES|EXPERTISE|'
    r'PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|EMPLOYMENT|WORK HISTORY|'
    r'EDUCATION|ACADEMIC BACKGROUND|EDUCATIONAL BACKGROUND|QUALIFICATIONS|'
    r'CERTIFICATIONS?|CERTIFICATES?|CREDENTIALS|LICENSES?|'
    r'PROJECTS?|KEY PROJECTS?|PERSONAL PROJECTS?|PORTFOLIO|'
    r'ACHIEVEMENTS?|ACCOMPLISHMENTS?|AWARDS?|HONORS?|'
    r'LANGUAGES?|PUBLICATIONS?|VOLUNTEER(?:ING)?|INTERESTS?)$',
    re.IGNORECASE
)

SECTION_MAP = {
    'summary':      {'PROFESSIONAL SUMMARY','SUMMARY','OBJECTIVE','CAREER OBJECTIVE','PROFILE','ABOUT ME'},
    'skills':       {'SKILLS','TECHNICAL SKILLS','KEY SKILLS','CORE COMPETENCIES','COMPETENCIES','EXPERTISE'},
    'experience':   {'PROFESSIONAL EXPERIENCE','WORK EXPERIENCE','EXPERIENCE','EMPLOYMENT','WORK HISTORY','EMPLOYMENT HISTORY'},
    'education':    {'EDUCATION','ACADEMIC BACKGROUND','EDUCATIONAL BACKGROUND','QUALIFICATIONS'},
    'certifications':{'CERTIFICATIONS','CERTIFICATION','CERTIFICATES','CERTIFICATE','CREDENTIALS','LICENSES','LICENSE'},
    'projects':     {'PROJECTS','PROJECT','KEY PROJECTS','PERSONAL PROJECTS','PORTFOLIO'},
    'achievements': {'ACHIEVEMENTS','ACHIEVEMENT','ACCOMPLISHMENTS','AWARDS','HONORS'},
    'languages':    {'LANGUAGES','LANGUAGE'},
    'other':        set()
}

def _get_section_key(header: str) -> str:
    upper = header.strip().upper()
    for key, variants in SECTION_MAP.items():
        if upper in variants:
            return key
    return 'other'


def parse_resume_sections(text: str) -> dict:
    """
    Parse resume text into sections dict.
    Returns {'header': str, 'summary': str, 'skills': str, 'experience': str, ...}
    """
    sections = {k: [] for k in SECTION_MAP}
    sections['header'] = []
    current = 'header'

    for line in text.split('\n'):
        stripped = line.strip()
        # Only match very short lines as headers
        if stripped and len(stripped) < 65 and not stripped.startswith('-') and SECTION_HEADER_RE.match(stripped):
            current = _get_section_key(stripped)
        else:
            sections[current].append(line)

    return {k: '\n'.join(v).strip() for k, v in sections.items()}


def extract_contact_info(header: str) -> dict:
    email_m   = re.search(r'[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}', header)
    phone_m   = re.search(r'(\+?\d[\d\s\-(). ]{7,15}\d)', header)
    linkedin_m= re.search(r'linkedin\.com/in/[\w\-]+', header, re.IGNORECASE)
    github_m  = re.search(r'github\.com/[\w\-]+', header, re.IGNORECASE)

    lines = [ln.strip() for ln in header.split('\n') if ln.strip()]
    name = lines[0] if lines else 'Your Name'

    return {
        'name':     name,
        'email':    email_m.group() if email_m else '',
        'phone':    phone_m.group().strip() if phone_m else '',
        'linkedin': linkedin_m.group() if linkedin_m else '',
        'github':   github_m.group() if github_m else '',
    }


# ─────────────────────────────────────────────────────────────
# 4. SKILL EXTRACTION & GAP ANALYSIS
# ─────────────────────────────────────────────────────────────

def extract_skills_from_text(text: str) -> set:
    """Extract skills from free text using substring matching against all role taxonomies."""
    text_lower = text.lower()
    found = set()

    # Collect every skill term from all roles
    all_terms = set()
    for role_data in ROLE_TAXONOMY.values():
        for s in role_data['required_skills'] + role_data['preferred_skills']:
            all_terms.add(s.lower())

    # Also add the ATS keywords
    for role_data in ROLE_TAXONOMY.values():
        for s in role_data['ats_keywords']:
            all_terms.add(s.lower())

    # Sorted by length descending to match multi-word terms first
    for term in sorted(all_terms, key=len, reverse=True):
        if term in text_lower:
            found.add(term)

    return found


def compute_skill_gap(resume_skills: set, role_key: str) -> dict:
    """
    Compare extracted resume skills against role taxonomy.
    Returns:
        strong:   skills present in resume AND are required/preferred
        partial:  semantically related but not exact match (TF-IDF)
        missing:  required skills not found at all
        score:    0-100 skill fit score
        coverage: % of required skills covered
    """
    taxonomy = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])
    required  = [s.lower() for s in taxonomy['required_skills']]
    preferred = [s.lower() for s in taxonomy['preferred_skills']]
    all_role_skills = set(required + preferred)

    resume_lower = {s.lower() for s in resume_skills}

    strong  = []
    missing_required = []
    missing_preferred = []

    for skill in required:
        if skill in resume_lower:
            strong.append(skill)
        else:
            # Check for partial match (substring)
            partial = any(skill in rs or rs in skill for rs in resume_lower)
            if partial:
                strong.append(skill)  # count as covered
            else:
                missing_required.append(skill)

    for skill in preferred:
        if skill in resume_lower:
            strong.append(skill)
        else:
            partial = any(skill in rs or rs in skill for rs in resume_lower)
            if not partial:
                missing_preferred.append(skill)

    # Use TF-IDF to find semantically similar skills for "partial" category
    partial = []
    if missing_required or missing_preferred:
        try:
            all_docs = list(resume_lower) + list(all_role_skills)
            if len(all_docs) > 2:
                vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
                tfidf_matrix = vect.fit_transform(all_docs)
                n_resume = len(resume_lower)
                resume_vecs = tfidf_matrix[:n_resume]
                for skill in missing_required[:5]:
                    try:
                        skill_vec = vect.transform([skill])
                        sims = cosine_similarity(skill_vec, resume_vecs).flatten()
                        if sims.max() > 0.4:
                            partial.append(skill)
                            missing_required = [s for s in missing_required if s != skill]
                    except Exception:
                        pass
        except Exception:
            pass

    # Score: required coverage weighted heavily
    req_covered = len(required) - len(missing_required)
    req_score   = (req_covered / max(len(required), 1)) * 70   # 70% weight
    pref_covered = len(preferred) - len(missing_preferred)
    pref_score   = (pref_covered / max(len(preferred), 1)) * 30  # 30% weight
    score = min(100, int(req_score + pref_score))

    coverage = round((req_covered / max(len(required), 1)) * 100, 1)

    return {
        'strong':            strong[:12],
        'partial':           partial[:5],
        'missing_required':  missing_required[:8],
        'missing_preferred': missing_preferred[:8],
        'score':             score,
        'coverage':          coverage,
        'role_title':        taxonomy['title']
    }


# ─────────────────────────────────────────────────────────────
# 5. SECTION SCORER
# ─────────────────────────────────────────────────────────────

def score_section(section_name: str, text: str, role_key: str, resume_skills: set) -> dict:
    """
    Score a resume section (0–100) with detailed NLP-based explanation.
    Returns {'score': int, 'grade': str, 'strengths': [...], 'weaknesses': [...], 'fixes': [...]}
    """
    if not text or len(text.strip()) < 10:
        return {
            'score': 0, 'grade': 'F',
            'strengths': [],
            'weaknesses': ['Section is missing or empty.'],
            'fixes': [f'Add a {section_name} section — it is essential for ATS and hiring managers.']
        }

    taxonomy  = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])
    lines     = [ln.strip() for ln in text.split('\n') if ln.strip()]
    text_lower= text.lower()
    score     = 50   # baseline
    strengths = []
    weaknesses= []
    fixes     = []

    if section_name == 'summary':
        words = text.split()
        wc = len(words)
        if wc >= 50:
            score += 15
            strengths.append(f'Good length ({wc} words) — substantive and readable.')
        elif wc >= 25:
            score += 5
            weaknesses.append(f'Summary is brief ({wc} words). Aim for 50–80 words.')
            fixes.append('Expand your summary to 50–80 words highlighting your key value proposition.')
        else:
            score -= 15
            weaknesses.append(f'Summary too short ({wc} words).')
            fixes.append('Rewrite summary with 50–80 words covering your role, top skills, and value.')

        # Check role alignment (key role terms in summary)
        role_terms_present = [t for t in taxonomy['ats_keywords'] if t in text_lower]
        if len(role_terms_present) >= 3:
            score += 15
            strengths.append(f'Role-aligned: mentions {len(role_terms_present)} key terms for {taxonomy["title"]}.')
        elif len(role_terms_present) >= 1:
            score += 5
            weaknesses.append('Summary mentions few role-specific keywords.')
            fixes.append(f'Add keywords like: {", ".join(taxonomy["ats_keywords"][:4])}')
        else:
            score -= 10
            weaknesses.append('Summary does not mention any role-specific terms.')
            fixes.append(f'Target your summary for {taxonomy["title"]} — include: {", ".join(taxonomy["ats_keywords"][:4])}')

        # Check for I-statements (ATS bad practice)
        if re.search(r'\bI\b', text):
            score -= 5
            weaknesses.append('Uses first-person "I" — avoid in resumes.')
            fixes.append('Remove all first-person pronouns. Start sentences with action verbs.')
        else:
            score += 5
            strengths.append('No first-person pronouns — professional tone.')

        # Metrics in summary
        if _count_metrics(text) > 0:
            score += 10
            strengths.append('Contains quantified achievement in summary — impressive.')
        else:
            weaknesses.append('No metrics in summary — add one quantified achievement.')
            fixes.append('Add a specific number: e.g., "with 5+ years experience delivering 30% efficiency gains".')

    elif section_name == 'experience':
        bullet_lines = [ln for ln in lines if len(ln) > 10]
        if not bullet_lines:
            return {'score': 10, 'grade': 'F',
                    'strengths': [], 'weaknesses': ['No experience bullets detected.'],
                    'fixes': ['Add bullet-point responsibilities and achievements under each role.']}

        # Count strong verbs
        strong_start_count = sum(1 for ln in bullet_lines if _starts_with_strong_verb(ln, taxonomy['action_verbs']))
        weak_start_count   = sum(1 for ln in bullet_lines if _starts_with_weak_verb(ln))
        passive_count      = sum(1 for ln in bullet_lines if _has_passive_voice(ln))
        metric_count       = sum(_count_metrics(ln) > 0 for ln in bullet_lines)
        total_bullets      = len(bullet_lines)

        strong_ratio = strong_start_count / max(total_bullets, 1)
        if strong_ratio >= 0.7:
            score += 20
            strengths.append(f'{strong_start_count}/{total_bullets} bullets start with strong verbs — excellent.')
        elif strong_ratio >= 0.4:
            score += 10
            weaknesses.append(f'Only {strong_start_count}/{total_bullets} bullets start with strong verbs.')
            fixes.append(f'Use verbs like: {", ".join(taxonomy["action_verbs"][:6])}')
        else:
            score -= 10
            weaknesses.append(f'Most bullets use weak or passive language ({weak_start_count} weak starts detected).')
            fixes.append(f'Rewrite bullets to start with: {", ".join(taxonomy["action_verbs"][:6])}')

        metric_ratio = metric_count / max(total_bullets, 1)
        if metric_ratio >= 0.5:
            score += 20
            strengths.append(f'{metric_count}/{total_bullets} bullets are quantified — very strong.')
        elif metric_ratio >= 0.25:
            score += 10
            weaknesses.append(f'Only {metric_count}/{total_bullets} bullets have metrics.')
            fixes.append('Add numbers, percentages, or dollar figures to at least half your bullets.')
        else:
            score -= 10
            weaknesses.append(f'Almost no quantified achievements ({metric_count} found).')
            fixes.append('Every bullet should answer "how much" or "how many" — add measurable results.')

        if passive_count > 0:
            score -= 5 * min(passive_count, 3)
            weaknesses.append(f'{passive_count} bullet(s) use passive voice.')
            fixes.append('Change passive to active: "Was responsible for X" → "Led X".')
        else:
            score += 5
            strengths.append('No passive voice detected — active writing throughout.')

        # Role keyword density in experience
        role_terms_in_exp = [t for t in taxonomy['ats_keywords'] if t in text_lower]
        if len(role_terms_in_exp) >= 4:
            score += 10
            strengths.append(f'Strong keyword density — {len(role_terms_in_exp)} role-specific terms in experience.')
        elif len(role_terms_in_exp) >= 2:
            score += 5
            weaknesses.append(f'Moderate keyword density — only {len(role_terms_in_exp)} role terms in experience.')
            fixes.append(f'Naturally weave in: {", ".join([t for t in taxonomy["ats_keywords"] if t not in text_lower][:3])}')
        else:
            score -= 5
            weaknesses.append('Very few role-specific keywords in experience section.')
            fixes.append(f'Include terms like: {", ".join(taxonomy["ats_keywords"][:5])}')

    elif section_name == 'skills':
        skill_matches = resume_skills & {s.lower() for s in taxonomy['required_skills'] + taxonomy['preferred_skills']}
        if len(skill_matches) >= 6:
            score += 25
            strengths.append(f'{len(skill_matches)} relevant skills detected — comprehensive.')
        elif len(skill_matches) >= 3:
            score += 10
            weaknesses.append(f'Only {len(skill_matches)} role-relevant skills found.')
            fixes.append(f'Add: {", ".join([s for s in taxonomy["required_skills"] if s.lower() not in {sk.lower() for sk in resume_skills}][:4])}')
        else:
            score -= 15
            weaknesses.append('Very few role-specific skills listed.')
            fixes.append(f'Required skills to add: {", ".join(taxonomy["required_skills"][:6])}')

        # Check for grouping / structure
        if ',' in text or '\n' in text.strip():
            score += 10
            strengths.append('Skills are structured (comma-separated or listed) — ATS-friendly.')
        else:
            weaknesses.append('Skills appear to be in a single block — structure them clearly.')
            fixes.append('Format skills as: "Category: Skill1, Skill2, Skill3"')

        # Check for soft vs hard skill balance
        soft_count = sum(1 for w in ['communication','teamwork','leadership','problem solving','time management'] if w in text_lower)
        hard_count = len(skill_matches)
        if hard_count > soft_count:
            score += 10
            strengths.append('Good balance — technical skills dominate over soft skills.')
        elif soft_count >= 3 and hard_count < 3:
            weaknesses.append('Too many soft skills, not enough hard/technical skills.')
            fixes.append('Prioritize hard skills and tools specific to your target role.')

    elif section_name == 'education':
        has_degree  = bool(re.search(r'\b(bachelor|master|phd|doctorate|b\.?sc?|m\.?sc?|b\.?e|m\.?e|mba|bba|b\.?tech|m\.?tech)\b', text_lower))
        has_year    = bool(re.search(r'\b(19|20)\d{2}\b', text))
        has_gpa     = bool(re.search(r'\b(gpa|cgpa|percentage|grade)\b', text_lower))
        has_courses = bool(re.search(r'\b(coursework|relevant courses|modules)\b', text_lower))

        if has_degree:
            score += 20
            strengths.append('Degree is clearly mentioned.')
        else:
            score -= 10
            weaknesses.append('Degree type not clearly identified.')
            fixes.append('Explicitly state your degree: e.g., "Bachelor of Science in Computer Science".')

        if has_year:
            score += 10
            strengths.append('Graduation year is present.')
        else:
            score -= 5
            weaknesses.append('Missing graduation year.')
            fixes.append('Add your graduation year (or expected graduation year).')

        if has_gpa:
            score += 5
            strengths.append('GPA/CGPA mentioned — adds credibility for recent grads.')

        if has_courses:
            score += 5
            strengths.append('Relevant coursework listed — useful for entry-level candidates.')

    # Clamp score
    score = max(10, min(100, score))
    if score >= 85:
        grade = 'A'
    elif score >= 70:
        grade = 'B'
    elif score >= 55:
        grade = 'C'
    elif score >= 40:
        grade = 'D'
    else:
        grade = 'F'

    return {
        'score':     score,
        'grade':     grade,
        'strengths': strengths,
        'weaknesses':weaknesses,
        'fixes':     fixes
    }


# ─────────────────────────────────────────────────────────────
# 6. SUMMARY GENERATOR (TF-IDF + Template Fusion)
# ─────────────────────────────────────────────────────────────

SUMMARY_TEMPLATES = {
    "DATA-SCIENCE": (
        "{trait} data professional with proven expertise in {skills}. "
        "Specializes in building end-to-end ML pipelines, surfacing actionable insights from complex datasets, "
        "and deploying models that directly drive business decisions. "
        "{achievement} "
        "Seeking a Data Scientist role where analytical precision meets real-world impact."
    ),
    "WEB-DEVELOPER": (
        "{trait} web developer skilled in {skills}, with a strong track record of building "
        "responsive, high-performance web applications. Expert in translating design requirements into "
        "clean, maintainable code across the full stack. "
        "{achievement} "
        "Looking to contribute as a Web Developer in a fast-paced, product-driven team."
    ),
    "SOFTWARE-ENGINEER": (
        "{trait} software engineer with deep expertise in {skills}. "
        "Experienced in designing scalable distributed systems, writing clean testable code, "
        "and leading end-to-end feature delivery in Agile environments. "
        "{achievement} "
        "Eager to solve hard engineering problems as a Software Engineer."
    ),
    "HR": (
        "{trait} HR professional experienced in {skills}. "
        "Skilled at building people-first programs — from talent acquisition and onboarding "
        "to performance management and culture development. "
        "{achievement} "
        "Committed to driving organizational growth through exceptional human resource practices."
    ),
    "DESIGNER": (
        "{trait} UI/UX designer with hands-on expertise in {skills}. "
        "Passionate about crafting intuitive, visually compelling experiences grounded in user research "
        "and data-informed design decisions. "
        "{achievement} "
        "Looking to join a design team that values creativity, usability, and measurable impact."
    ),
    "FINANCE": (
        "{trait} finance professional with expertise in {skills}. "
        "Skilled at financial modeling, budgeting, and analysis that supports strategic C-suite decisions. "
        "Proven ability to interpret complex data and translate it into actionable financial insights. "
        "{achievement} "
        "Seeking a Finance role where precision and strategy intersect."
    ),
    "INFORMATION-TECHNOLOGY": (
        "{trait} IT professional with extensive experience in {skills}. "
        "Track record of managing robust infrastructure, resolving critical incidents, "
        "and implementing secure, scalable technology solutions. "
        "{achievement} "
        "Driven to keep systems running reliably while advancing IT maturity."
    ),
    "SALES": (
        "{trait} sales professional with a consistent record of exceeding targets through {skills}. "
        "Expert at building lasting client relationships, identifying growth opportunities, "
        "and closing deals in competitive markets. "
        "{achievement} "
        "Ready to bring hunter energy and strategic relationship skills to a high-performance sales team."
    ),
    "HEALTHCARE": (
        "{trait} healthcare professional with clinical expertise in {skills}. "
        "Dedicated to delivering the highest standard of patient care, maintaining rigorous compliance, "
        "and working effectively in high-pressure medical environments. "
        "{achievement} "
        "Committed to improving patient outcomes through evidence-based practice."
    ),
    "ENGINEERING": (
        "{trait} engineer with hands-on expertise in {skills}. "
        "Experienced in delivering engineering projects from concept to completion — on time, "
        "within budget, and to the highest quality standards. "
        "{achievement} "
        "Looking to bring technical precision and a continuous-improvement mindset to complex engineering challenges."
    ),
    "ACCOUNTANT": (
        "{trait} accounting professional with deep expertise in {skills}. "
        "Skilled in maintaining 100% accurate financial records, ensuring regulatory compliance, "
        "and streamlining accounting processes for maximum efficiency. "
        "{achievement} "
        "Seeking an Accounting role where meticulous attention to detail and financial integrity matter."
    ),
    "BANKING": (
        "{trait} banking professional with experience in {skills}. "
        "Strong background in regulatory compliance, credit analysis, and building long-term "
        "client relationships that drive portfolio growth. "
        "{achievement} "
        "Committed to delivering excellent banking service with full regulatory adherence."
    ),
    "BUSINESS-DEVELOPMENT": (
        "{trait} business development professional with expertise in {skills}. "
        "Skilled at identifying high-value growth opportunities, forging strategic partnerships, "
        "and driving measurable revenue in competitive markets. "
        "{achievement} "
        "Ready to bring a hunter mindset and strategic vision to a BD role that rewards results."
    ),
    "DIGITAL-MEDIA": (
        "{trait} digital media professional with expertise in {skills}. "
        "Proven track record of developing data-driven content strategies that grow brand presence, "
        "drive engagement, and deliver measurable campaign ROI. "
        "{achievement} "
        "Looking to create impactful digital narratives for a brand that values creativity and performance."
    ),
    "DEFAULT": (
        "{trait} professional with expertise in {skills}. "
        "Proven ability to deliver high-quality work, collaborate effectively across teams, "
        "and drive measurable results in dynamic environments. "
        "{achievement} "
        "Eager to contribute meaningfully in a challenging and growth-oriented role."
    )
}


def generate_role_summary(role_key: str, resume_skills: set, sections: dict, contact: dict) -> str:
    """
    Generate a human-sounding, role-specific professional summary using:
    1. TF-IDF to extract most relevant phrases from the resume
    2. Skill filtering to role-relevant terms
    3. Template fusion with trait + achievement injection
    """
    taxonomy = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])

    # Select top skills relevant to this role
    role_required_lower = {s.lower() for s in taxonomy['required_skills']}
    role_preferred_lower = {s.lower() for s in taxonomy['preferred_skills']}
    all_role_skills = role_required_lower | role_preferred_lower

    matched = [s for s in resume_skills if s.lower() in all_role_skills]
    # Also add skills that are substrings of role skills
    for rs in resume_skills:
        for rsk in all_role_skills:
            if rs.lower() in rsk or rsk in rs.lower():
                if rs not in matched:
                    matched.append(rs)

    # Use proper capitalization from taxonomy
    skill_display = []
    for s in matched[:5]:
        # Find the properly capitalized version from taxonomy
        for ts in taxonomy['required_skills'] + taxonomy['preferred_skills']:
            if s.lower() == ts.lower():
                skill_display.append(ts)
                break
        else:
            skill_display.append(s.title() if len(s) <= 4 else s.capitalize())

    # Fallback: if no matched skills, use top required skills
    if not skill_display:
        skill_display = taxonomy['required_skills'][:4]

    skills_str = ', '.join(skill_display[:4])

    # Pick a random trait that fits the role
    trait = random.choice(taxonomy['summary_traits']).capitalize()

    # Pick an achievement pattern and fill with plausible numbers
    pattern = random.choice(taxonomy['achievement_patterns'])
    # Replace {n} tokens with plausible numbers
    achievement = re.sub(
        r'\{n\}',
        lambda m: str(random.choice([10, 15, 20, 25, 30, 35, 40, 50])),
        pattern
    )

    template = SUMMARY_TEMPLATES.get(role_key, SUMMARY_TEMPLATES['DEFAULT'])
    summary = template.format(
        trait=trait,
        skills=skills_str,
        achievement=achievement
    )

    # Check if resume has an existing summary with good fragments to incorporate
    existing = sections.get('summary', '').strip()
    if existing and len(existing) > 30:
        # TF-IDF: extract the most relevant sentence from existing summary
        try:
            sents = re.split(r'(?<=[.!?])\s+', existing)
            if len(sents) >= 2:
                ref = [taxonomy['title']] + taxonomy['ats_keywords'][:5]
                vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                corpus = sents + ref
                tfidf  = vect.fit_transform(corpus)
                ref_vec = tfidf[-len(ref):]
                sent_vecs = tfidf[:len(sents)]
                sims = cosine_similarity(sent_vecs, ref_vec).mean(axis=1)
                best_sent = sents[int(np.argmax(sims))].strip()
                if len(best_sent) > 20 and best_sent.lower() not in summary.lower():
                    summary = summary + ' ' + best_sent
        except Exception:
            pass

    # Break into ATS-safe line lengths (no line > 35 words)
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    output_lines = []
    current_line = []
    current_count = 0
    for sent in sentences:
        wc = len(sent.split())
        if current_count + wc > 35 and current_line:
            output_lines.append(' '.join(current_line))
            current_line = [sent]
            current_count = wc
        else:
            current_line.append(sent)
            current_count += wc
    if current_line:
        output_lines.append(' '.join(current_line))

    return '\n'.join(output_lines)


# ─────────────────────────────────────────────────────────────
# 7. EXPERIENCE REWRITER
# ─────────────────────────────────────────────────────────────

def rewrite_experience_section(exp_text: str, role_key: str) -> tuple:
    """
    Rewrite all bullets in the experience section.
    Returns (rewritten_text, total_changes_made, change_log)
    """
    taxonomy = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])
    role_verbs = taxonomy['action_verbs']

    lines = exp_text.split('\n')
    rewritten_lines = []
    total_changes = 0
    change_log = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            rewritten_lines.append(line)
            continue

        # Detect if this is a bullet line vs a header/date line
        is_bullet = (
            stripped.startswith(('-', '•', '●', '▪', '◦', '◆', '*')) or
            (len(stripped.split()) >= 5 and not re.match(r'^[A-Z][A-Z\s,.\-&]+$', stripped) and
             not re.match(r'^(19|20)\d{2}', stripped))
        )

        if is_bullet and len(stripped) >= 10:
            rewritten, changes = _rewrite_bullet(stripped, role_verbs, role_key)
            if changes:
                total_changes += len(changes)
                change_log.extend(changes)
                rewritten_lines.append(rewritten)
            else:
                rewritten_lines.append(stripped)
        else:
            rewritten_lines.append(line)

    return '\n'.join(rewritten_lines), total_changes, change_log


# ─────────────────────────────────────────────────────────────
# 8. SKILLS SECTION BUILDER
# ─────────────────────────────────────────────────────────────

def build_skills_section(role_key: str, resume_skills: set, existing_skills_text: str) -> str:
    """
    Build a structured, role-targeted skills section.
    Categorizes skills into: Core Technical | Tools & Platforms | Soft Skills
    Only lists skills present in the resume + must-have role skills if absent.
    """
    taxonomy = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])

    required_lower  = {s.lower(): s for s in taxonomy['required_skills']}
    preferred_lower = {s.lower(): s for s in taxonomy['preferred_skills']}

    core_tech   = []
    tools_plat  = []
    soft_skills = []
    seen = set()

    soft_kw = {'communication','teamwork','leadership','problem solving','critical thinking',
               'time management','negotiation','presentation','collaboration','adaptability',
               'attention to detail','organizational skills','planning','reporting','creativity',
               'empathy','conflict resolution','customer service'}

    tool_kw = {'excel','powerpoint','microsoft office','word','outlook','jira','confluence',
               'trello','slack','salesforce','sap','erp','hris','tableau','power bi',
               'google analytics','hubspot','wordpress','figma','sketch','adobe'}

    # 1. Skills from resume matched to role
    for skill in resume_skills:
        s_lower = skill.lower()
        if s_lower in seen:
            continue

        # Get display form
        if s_lower in required_lower:
            display = required_lower[s_lower]
        elif s_lower in preferred_lower:
            display = preferred_lower[s_lower]
        else:
            display = skill.title() if len(skill) <= 4 else skill.capitalize()

        if s_lower in soft_kw:
            soft_skills.append(display)
        elif any(t in s_lower for t in tool_kw):
            tools_plat.append(display)
        elif s_lower in required_lower or s_lower in preferred_lower:
            core_tech.append(display)
        seen.add(s_lower)

    # 2. Ensure all required skills appear (pad with role requirements if missing)
    for s_lower, display in required_lower.items():
        if s_lower not in seen and len(core_tech) < 10:
            core_tech.append(display)
            seen.add(s_lower)

    # Build output
    parts = []
    if core_tech:
        parts.append(f"Core: {', '.join(core_tech[:8])}")
    if tools_plat:
        parts.append(f"Tools: {', '.join(tools_plat[:6])}")
    if soft_skills:
        parts.append(f"Soft Skills: {', '.join(soft_skills[:5])}")

    if not parts:
        # Minimal fallback
        parts = [', '.join([s for s in taxonomy['required_skills'][:8]])]

    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────
# 9. ATS COMPLIANCE CHECKER
# ─────────────────────────────────────────────────────────────

def run_ats_check(text: str, role_key: str, resume_skills: set) -> dict:
    """
    Run comprehensive ATS compliance check and return score + issues.
    """
    taxonomy  = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])
    text_lower= text.lower()
    issues    = []
    passed    = []
    score     = 100

    # 1. Word count
    wc = len(text.split())
    if wc < 250:
        issues.append(f"Too short ({wc} words). Target 350–700 words.")
        score -= 20
    elif wc > 900:
        issues.append(f"Too long ({wc} words). Trim to under 750 words.")
        score -= 10
    else:
        passed.append(f"Word count optimal ({wc} words).")

    # 2. Contact info
    has_email = bool(re.search(r'[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}', text))
    has_phone = bool(re.search(r'(\+?\d[\d\s\-(). ]{7,15}\d)', text))
    if not has_email:
        issues.append("No email address detected.")
        score -= 15
    else:
        passed.append("Email present.")
    if not has_phone:
        issues.append("No phone number detected.")
        score -= 10
    else:
        passed.append("Phone number present.")

    # 3. Required sections
    section_checks = {
        'Summary':    r'\b(summary|objective|profile|professional summary)\b',
        'Experience': r'\b(experience|employment|work history)\b',
        'Skills':     r'\b(skills|competencies|technical skills)\b',
        'Education':  r'\b(education|degree|university|college)\b',
    }
    for section, pattern in section_checks.items():
        if re.search(pattern, text_lower):
            passed.append(f"{section} section present.")
        else:
            issues.append(f"Missing {section} section.")
            score -= 10

    # 4. ATS keyword density
    kw_hits = [kw for kw in taxonomy['ats_keywords'] if kw.lower() in text_lower]
    kw_coverage = len(kw_hits) / max(len(taxonomy['ats_keywords']), 1)
    if kw_coverage >= 0.6:
        passed.append(f"Strong keyword density ({len(kw_hits)}/{len(taxonomy['ats_keywords'])} role keywords).")
    elif kw_coverage >= 0.3:
        issues.append(f"Moderate keywords ({len(kw_hits)}/{len(taxonomy['ats_keywords'])}). Add: {', '.join([k for k in taxonomy['ats_keywords'] if k not in kw_hits][:3])}")
        score -= 10
    else:
        issues.append(f"Low keyword density. Add: {', '.join(taxonomy['ats_keywords'][:5])}")
        score -= 20

    # 5. Action verbs
    role_verb_hits = [v for v in taxonomy['action_verbs'] if v in text_lower]
    if len(role_verb_hits) >= 6:
        passed.append(f"Strong action verb usage ({len(role_verb_hits)} detected).")
    elif len(role_verb_hits) >= 3:
        issues.append(f"More action verbs needed. Try: {', '.join([v for v in taxonomy['action_verbs'] if v not in role_verb_hits][:4])}")
        score -= 5
    else:
        issues.append(f"Very few role-specific action verbs. Use: {', '.join(taxonomy['action_verbs'][:6])}")
        score -= 15

    # 6. Quantification
    metric_count = _count_metrics(text)
    if metric_count >= 5:
        passed.append(f"Well-quantified ({metric_count} metrics detected).")
    elif metric_count >= 2:
        issues.append(f"Add more metrics. Only {metric_count} quantified achievements found.")
        score -= 5
    else:
        issues.append("Almost no quantified achievements. Add numbers, percentages, dollar figures.")
        score -= 15

    # 7. Special characters / ATS killers
    special_ratio = len(re.findall(r'[^\w\s.,;:!?()\-\'/\n@+%$]', text)) / max(len(text), 1)
    if special_ratio < 0.01:
        passed.append("Clean formatting — no problematic special characters.")
    elif special_ratio < 0.03:
        issues.append("Some special characters detected. ATS may misread tables or symbols.")
        score -= 5
    else:
        issues.append("Many special characters detected — simplify formatting.")
        score -= 15

    # 8. Line length (no 50+ word lines)
    long_lines = [ln for ln in text.split('\n') if len(ln.split()) > 50]
    if long_lines:
        issues.append(f"{len(long_lines)} very long line(s) detected. Break into bullet points.")
        score -= 10
    else:
        passed.append("Line lengths are ATS-safe (no overly long paragraphs).")

    # 9. First-person
    if re.search(r'\bI\b', text):
        issues.append("First-person 'I' found. Remove for professional tone.")
        score -= 5
    else:
        passed.append("No first-person pronouns.")

    score = max(0, min(100, score))
    if score >= 90:
        grade = "Excellent"
    elif score >= 80:
        grade = "Good"
    elif score >= 65:
        grade = "Fair"
    elif score >= 50:
        grade = "Poor"
    else:
        grade = "Critical"

    return {
        'score':  score,
        'grade':  grade,
        'passed': passed,
        'issues': issues,
        'keyword_coverage': round(kw_coverage * 100, 1),
        'keywords_found':   kw_hits,
        'keywords_missing': [k for k in taxonomy['ats_keywords'] if k not in kw_hits]
    }


# ─────────────────────────────────────────────────────────────
# 10. MASTER RESUME ASSEMBLER
# ─────────────────────────────────────────────────────────────

SECTION_DIVIDER = '─' * 48


def assemble_resume(contact: dict, role_key: str, summary: str,
                    skills_text: str, exp_text: str,
                    sections: dict) -> str:
    """Assemble the final ATS-optimized resume text."""
    taxonomy = ROLE_TAXONOMY.get(role_key, ROLE_TAXONOMY['DEFAULT'])
    role_display = taxonomy['title']

    parts = []
    parts.append(contact['name'].upper())

    contact_parts = list(filter(None, [contact['email'], contact['phone'],
                                        contact['linkedin'], contact['github']]))
    if contact_parts:
        parts.append(' | '.join(contact_parts))
    parts.append(f'Targeted for: {role_display}')
    parts.append('')

    parts.append('PROFESSIONAL SUMMARY')
    parts.append(SECTION_DIVIDER)
    parts.append(summary)
    parts.append('')

    parts.append('SKILLS')
    parts.append(SECTION_DIVIDER)
    parts.append(skills_text)
    parts.append('')

    if exp_text.strip():
        parts.append('PROFESSIONAL EXPERIENCE')
        parts.append(SECTION_DIVIDER)
        parts.append(exp_text.strip())
        parts.append('')

    if sections.get('projects', '').strip():
        parts.append('PROJECTS')
        parts.append(SECTION_DIVIDER)
        parts.append(sections['projects'].strip())
        parts.append('')

    if sections.get('education', '').strip():
        parts.append('EDUCATION')
        parts.append(SECTION_DIVIDER)
        parts.append(sections['education'].strip())
        parts.append('')

    if sections.get('certifications', '').strip():
        parts.append('CERTIFICATIONS')
        parts.append(SECTION_DIVIDER)
        parts.append(sections['certifications'].strip())
        parts.append('')

    if sections.get('achievements', '').strip():
        parts.append('ACHIEVEMENTS')
        parts.append(SECTION_DIVIDER)
        parts.append(sections['achievements'].strip())
        parts.append('')

    if sections.get('languages', '').strip():
        parts.append('LANGUAGES')
        parts.append(SECTION_DIVIDER)
        parts.append(sections['languages'].strip())
        parts.append('')

    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────
# 11. MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────

def enhance_resume_for_role(resume_text: str, target_role: str) -> dict:
    """
    Master function. Given raw resume text and a target role string,
    returns a rich enhancement result dict:
    {
        'enhanced_resume':    str,        # Full rewritten resume text
        'skill_gap':          dict,       # strong/partial/missing/score/coverage
        'section_scores':     dict,       # per-section score + grade + strengths/weaknesses
        'ats_result':         dict,       # ATS compliance score + issues
        'summary':            str,        # new summary text
        'skills_text':        str,        # new skills section
        'experience_changes': int,        # number of bullets improved
        'change_log':         list,       # what NLP changed
        'role_key':           str,
        'role_title':         str,
        'contact':            dict,
    }
    """
    # Normalize role key
    role_upper = target_role.upper().replace(' ', '-').replace('_', '-')
    if role_upper in ROLE_TAXONOMY:
        role_key = role_upper
    else:
        # Fuzzy match
        role_key = 'DEFAULT'
        for key in ROLE_TAXONOMY:
            if key in role_upper or role_upper in key:
                role_key = key
                break

    # 1. Parse sections
    sections = parse_resume_sections(resume_text)
    contact  = extract_contact_info(sections.get('header', resume_text[:300]))

    # 2. Extract skills from entire resume text
    resume_skills = extract_skills_from_text(resume_text)

    # 3. Skill gap analysis
    skill_gap = compute_skill_gap(resume_skills, role_key)

    # 4. Score each section (BEFORE enhancement)
    section_scores = {
        'summary':    score_section('summary', sections.get('summary', ''), role_key, resume_skills),
        'experience': score_section('experience', sections.get('experience', ''), role_key, resume_skills),
        'skills':     score_section('skills', sections.get('skills', ''), role_key, resume_skills),
        'education':  score_section('education', sections.get('education', ''), role_key, resume_skills),
    }

    # 5. Generate new summary
    new_summary = generate_role_summary(role_key, resume_skills, sections, contact)

    # 6. Rewrite experience bullets
    exp_raw = sections.get('experience', '').strip()
    if exp_raw:
        new_exp, exp_changes, change_log = rewrite_experience_section(exp_raw, role_key)
    else:
        new_exp, exp_changes, change_log = '', 0, []

    # 7. Build skills section
    existing_skills = sections.get('skills', '')
    new_skills = build_skills_section(role_key, resume_skills, existing_skills)

    # 8. Assemble enhanced resume
    enhanced_text = assemble_resume(
        contact, role_key, new_summary, new_skills, new_exp, sections
    )

    # 9. ATS check on enhanced resume
    ats_result = run_ats_check(enhanced_text, role_key, resume_skills)

    # 10. Compute overall enhancement stats
    before_metrics = _count_metrics(resume_text)
    after_metrics  = _count_metrics(enhanced_text)

    return {
        'enhanced_resume':    enhanced_text,
        'skill_gap':          skill_gap,
        'section_scores':     section_scores,
        'ats_result':         ats_result,
        'summary':            new_summary,
        'skills_text':        new_skills,
        'experience_changes': exp_changes,
        'change_log':         list(set(change_log)),
        'role_key':           role_key,
        'role_title':         ROLE_TAXONOMY[role_key]['title'],
        'contact':            contact,
        'stats': {
            'metrics_before': before_metrics,
            'metrics_after':  after_metrics,
            'skills_matched': len(skill_gap['strong']),
            'skills_missing': len(skill_gap['missing_required']),
            'ats_score':      ats_result['score'],
            'bullets_improved': exp_changes,
        }
    }