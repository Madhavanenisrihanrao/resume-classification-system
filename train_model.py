"""
train_model.py
==============
Trains a TF-IDF + LinearSVC pipeline on the UpdatedResumeDataSet.
Run this ONCE to generate resume_classifier.pkl and label_encoder.pkl

Usage:
    python train_model.py
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ── 1. Download NLTK data ────────────────────────────────────────────────────
print("[*] Downloading NLTK data...")
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ── 2. Resume categories & sample data ─────────────────────────────────────
CATEGORIES = {
    "Data Science": [
        "experience data scientist machine learning python pandas numpy scikit-learn "
        "statistical analysis data visualization matplotlib seaborn predictive modeling "
        "feature engineering cross-validation hyperparameter tuning sql database",
        "data science professional statistical modeling deep learning tensorflow keras "
        "neural networks regression classification clustering pca dimensionality reduction "
        "jupyter notebook github version control business intelligence tableau power bi",
        "senior data scientist nlp natural language processing bert transformers huggingface "
        "time series forecasting arima prophet a/b testing hypothesis testing statistics "
        "r programming ggplot2 data wrangling etl pipeline aws s3 spark",
        "data analyst python sql tableau power bi excel statistics data visualization "
        "business intelligence reporting dashboards kpi metrics data cleaning exploration "
        "machine learning basics logistic regression decision tree random forest",
        "data science intern kaggle competitions feature engineering ensemble methods "
        "gradient boosting xgboost lightgbm neural networks computer vision image classification",
    ],
    "Machine Learning": [
        "machine learning engineer deep learning pytorch tensorflow computer vision "
        "convolutional neural networks object detection yolo resnet vgg image segmentation "
        "transfer learning model deployment mlops docker kubernetes",
        "ml engineer reinforcement learning policy gradient q-learning openai gym "
        "natural language processing text classification sentiment analysis bert gpt "
        "recommendation systems collaborative filtering matrix factorization",
        "machine learning researcher graph neural networks attention mechanisms "
        "generative adversarial networks variational autoencoders contrastive learning "
        "few-shot learning meta-learning bayesian optimization hyperparameter search",
        "applied ml engineer production machine learning model serving fastapi flask "
        "feature store monitoring drift detection data pipeline airflow mlflow "
        "experiment tracking model registry canary deployment shadow mode",
        "ml ops engineer infrastructure machine learning ci cd pipelines model training "
        "distributed training gpu cuda mixed precision quantization pruning onnx triton",
    ],
    "Web Development": [
        "web developer react javascript typescript nodejs express mongodb html css "
        "responsive design redux state management rest api graphql authentication jwt "
        "git version control agile scrum frontend backend fullstack",
        "frontend developer vue angular react hooks context api webpack babel "
        "sass scss tailwind css animations performance optimization seo accessibility "
        "testing jest cypress unit integration e2e deployment vercel netlify",
        "backend developer python django flask postgresql mysql redis celery "
        "microservices docker kubernetes aws azure gcp rest graphql api design "
        "authentication oauth jwt security cors rate limiting caching",
        "fullstack developer nextjs react nodejs express postgresql prisma graphql "
        "typescript tailwind vercel deployment ci cd github actions testing vitest "
        "websockets real-time applications authentication stripe payments",
        "web developer php laravel symfony mysql wordpress woocommerce javascript "
        "jquery bootstrap responsive design seo google analytics email marketing "
        "cpanel hosting domain management security ssl maintenance",
    ],
    "Java Developer": [
        "java developer spring boot microservices hibernate jpa postgresql mysql "
        "maven gradle junit mockito rest api docker kubernetes aws jenkins ci cd "
        "design patterns solid principles clean code agile scrum",
        "senior java engineer spring framework spring cloud netflix oss kafka "
        "rabbitmq redis distributed systems high availability fault tolerance "
        "load balancing circuit breaker eureka zuul api gateway oauth2",
        "java backend developer multithreading concurrency collections streams "
        "lambda functional programming jvm optimization garbage collection "
        "profiling monitoring actuator prometheus grafana elk stack logging",
        "android developer java kotlin jetpack compose android sdk room database "
        "retrofit okhttp firebase google play services material design mvvm "
        "architecture components navigation lifecycle viewmodel livedata",
        "java enterprise developer jee ejb jsf servlet jsp jms weblogic "
        "websphere oracle database pl sql stored procedures xml xpath xslt "
        "soap web services wsdl jax-ws integration middleware",
    ],
    "Python Developer": [
        "python developer django rest framework postgresql redis celery "
        "asyncio websockets pytest coverage type hints mypy black linting "
        "docker docker-compose kubernetes helm aws lambda serverless",
        "python engineer fastapi pydantic sqlalchemy alembic postgresql "
        "redis kafka event-driven architecture microservices grpc protocol buffers "
        "opentelemetry monitoring distributed tracing sentry logging",
        "python developer automation scripting bash selenium playwright "
        "web scraping beautifulsoup scrapy api integration data processing "
        "pandas numpy file handling csv json xml etl pipelines",
    ],
    "DevOps Engineer": [
        "devops engineer aws azure gcp terraform ansible chef puppet "
        "kubernetes docker jenkins gitlab ci github actions infrastructure as code "
        "monitoring prometheus grafana elk alertmanager pagerduty",
        "site reliability engineer sre linux shell scripting python go "
        "incident management postmortem runbooks service level objectives "
        "error budgets toil reduction chaos engineering",
    ],
    "HR": [
        "human resources hr manager talent acquisition recruitment onboarding "
        "performance management employee relations compensation benefits "
        "hris workday successfactors payroll compliance labor law",
        "hr business partner organizational development training learning "
        "succession planning workforce planning diversity inclusion "
        "employee engagement culture change management",
    ],
    "Advocate": [
        "lawyer attorney litigation corporate law mergers acquisitions "
        "contract drafting legal research due diligence compliance "
        "dispute resolution arbitration mediation court appearances",
        "legal counsel intellectual property patents trademarks copyright "
        "licensing agreements employment law real estate transactions "
        "regulatory compliance corporate governance",
    ],
    "Arts": [
        "graphic designer adobe photoshop illustrator indesign figma sketch "
        "ui ux design visual identity branding logo design typography "
        "color theory print digital media motion graphics",
        "artist painter sculptor installation art exhibition gallery "
        "contemporary art fine arts mixed media creative direction "
        "art direction concept development visual storytelling",
    ],
    "Mechanical Engineer": [
        "mechanical engineer solidworks autocad catia ansys simulation "
        "finite element analysis product design manufacturing process "
        "thermodynamics fluid mechanics materials science cnc machining",
        "manufacturing engineer lean six sigma quality control "
        "production planning supply chain management gd&t tolerance "
        "process improvement kaizen 5s statistical process control",
    ],
    "Sales": [
        "sales manager b2b saas crm salesforce hubspot pipeline management "
        "account executive business development lead generation "
        "quota attainment revenue growth strategic partnerships",
        "sales representative cold calling prospecting discovery calls "
        "demos negotiation closing objection handling customer success "
        "upselling cross-selling relationship management",
    ],
    "Health and Fitness": [
        "personal trainer certified fitness instructor strength conditioning "
        "nutrition coaching body composition weight loss athletic performance "
        "gym management group fitness yoga pilates",
        "healthcare professional nurse practitioner clinical assessment "
        "patient care medication management electronic health records "
        "telehealth preventive care wellness programs",
    ],
    "Civil Engineer": [
        "civil engineer structural design autocad revit staad pro "
        "construction management project planning rcc design "
        "geotechnical soil mechanics highway bridge design",
        "site engineer quantity surveying cost estimation BoQ "
        "construction supervision quality assurance safety management "
        "contract administration project coordination",
    ],
    "Business Analyst": [
        "business analyst requirements gathering stakeholder management "
        "process mapping bpmn uml use cases user stories agile "
        "data analysis sql tableau power bi jira confluence",
        "product analyst market research competitive analysis "
        "product roadmap kpi metrics ab testing user research "
        "wireframing prototyping figma business intelligence",
    ],
    "Digital Marketing": [
        "digital marketing seo sem google ads facebook ads instagram "
        "content marketing email marketing hubspot mailchimp analytics "
        "google analytics conversion optimization cro landing pages",
        "social media manager community management content creation "
        "influencer marketing brand awareness campaigns performance "
        "marketing paid social tiktok linkedin twitter engagement",
    ],
    "Testing": [
        "qa engineer software testing manual testing automation selenium "
        "appium pytest junit testng test planning test cases defect tracking "
        "jira agile regression smoke integration performance testing",
        "test automation engineer cypress playwright selenium webdriver "
        "api testing postman rest assured bdd cucumber gherkin "
        "ci cd jenkins code coverage test reporting",
    ],
    "PMO": [
        "project manager pmp agile scrum kanban waterfall stakeholder "
        "risk management budget planning resource allocation jira "
        "confluence ms project roadmap scheduling delivery",
        "program manager portfolio management strategic planning "
        "governance reporting executive communication change management "
        "benefits realization organizational transformation",
    ],
    "ETL Developer": [
        "etl developer informatica talend ssis datastage data warehouse "
        "dimensional modeling star schema snowflake fact dimension "
        "sql stored procedures data migration integration",
        "data engineer apache spark hadoop hive kafka airflow "
        "dbt snowflake bigquery redshift pipeline orchestration "
        "streaming batch processing data lake governance",
    ],
    "Operations Manager": [
        "operations manager supply chain logistics inventory management "
        "warehouse operations process optimization vendor management "
        "kpi reporting team leadership cost reduction efficiency",
        "operations analyst process improvement lean methodology "
        "erp sap oracle workflow automation reporting dashboards "
        "cross-functional collaboration operational excellence",
    ],
    "Network Security Engineer": [
        "network security engineer firewall vpn intrusion detection "
        "penetration testing vulnerability assessment cisco fortinet "
        "siem splunk threat intelligence incident response",
        "cybersecurity analyst soc nist framework iso 27001 "
        "endpoint security identity access management pam "
        "zero trust architecture cloud security aws azure security",
    ],
    "Blockchain": [
        "blockchain developer solidity ethereum smart contracts "
        "web3 hardhat truffle metamask defi protocols uniswap "
        "nft erc20 erc721 ipfs decentralized applications dapps",
        "blockchain engineer hyperledger fabric chaincode go "
        "consensus mechanisms proof of work stake distributed ledger "
        "cryptocurrency bitcoin lightning network layer 2 scaling",
    ],
    "Database": [
        "database administrator postgresql mysql oracle mongodb "
        "performance tuning query optimization indexing partitioning "
        "backup recovery high availability replication clustering",
        "data architect database design erd normalization "
        "nosql cassandra redis elasticsearch couchdb document "
        "time series influxdb graph databases neo4j",
    ],
}


def clean_text(text: str) -> str:
    """Clean and normalize resume text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def build_dataset() -> pd.DataFrame:
    """Build training dataset from synthetic category examples."""
    print("[*] Building dataset...")
    records = []
    for category, samples in CATEGORIES.items():
        for sample in samples:
            # Augment each sample with slight variations
            records.append({"Resume": sample, "Category": category})
            records.append({"Resume": sample + " " + sample[:len(sample)//2], "Category": category})

    df = pd.DataFrame(records)
    df["Clean_Resume"] = df["Resume"].apply(clean_text)
    print(f"    ✓ Dataset: {len(df)} samples across {df['Category'].nunique()} categories")
    print(f"    Categories: {sorted(df['Category'].unique())}")
    return df


def train(df: pd.DataFrame):
    """Train pipeline and save artifacts."""
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    X = df["Clean_Resume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use raw LinearSVC to expose decision_function for temperature-scaled softmax
    clf_svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", clf_svc),
    ])

    print("\n[*] Training pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n    ✓ Test Accuracy: {acc:.4f}")
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names  = le.classes_[present_labels]
    print("\n" + classification_report(y_test, y_pred,
                                       labels=present_labels,
                                       target_names=present_names,
                                       zero_division=0))

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"    ✓ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline, le, X_test, y_test, y_pred


def save_artifacts(pipeline, le):
    """Persist model and encoder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "resume_classifier.pkl"), "wb") as f:
        pickle.dump(pipeline, f)
    with open(os.path.join(base_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print("\n    ✓ Saved resume_classifier.pkl")
    print("    ✓ Saved label_encoder.pkl")


def plot_confusion_matrix(le, y_test, y_pred):
    """Save confusion matrix chart."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Resume Classifier", fontsize=14, pad=15)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "confusion_matrix.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"    ✓ Saved confusion matrix → {out_path}")


if __name__ == "__main__":
    df = build_dataset()
    pipeline, le, X_test, y_test, y_pred = train(df)
    save_artifacts(pipeline, le)
    plot_confusion_matrix(le, y_test, y_pred)
    print("\n✅ Training complete! Run `python app.py` to start the web server.\n")
