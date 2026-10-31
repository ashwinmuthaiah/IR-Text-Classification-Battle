from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, FIG = ROOT/"data"/"raw", ROOT/"outputs", ROOT/"figures"
SEED, PER_CLASS, MIN_CLASS = 42, 10000, 1000

def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)

def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    files = list(RAW.glob("*.json")) + list(RAW.glob("*.jsonl"))
    if not files:
        raise SystemExit("Dataset not found. Download News_Category_Dataset_v3.json from Kaggle and put it in data/raw/.")
    df = load_jsonl(files[0])
    required = {"category", "headline"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Missing expected columns. Found: {list(df.columns)}")
    desc = df["short_description"].fillna("").astype(str) if "short_description" in df else ""
    df["text"] = df["headline"].fillna("").astype(str) + " " + desc
    df["label"] = df["category"].astype(str).str.strip()
    df = df[df.text.str.strip().ne("") & df.label.str.strip().ne("")]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    all_counts = df.label.value_counts()
    all_counts.rename_axis("category").reset_index(name="count").to_csv(OUT/"full_class_distribution.csv", index=False)
    chosen = all_counts[all_counts >= MIN_CLASS].head(4).index.tolist()
    if len(chosen) < 4:
        raise SystemExit("Fewer than four categories meet the minimum count. Inspect full_class_distribution.csv.")
    pieces = []
    for label in chosen:
        group = df[df.label == label]
        pieces.append(group.sample(n=min(PER_CLASS, len(group)), random_state=SEED))
    data = pd.concat(pieces).sample(frac=1, random_state=SEED).reset_index(drop=True)
    dist = data.label.value_counts().rename_axis("category").reset_index(name="count")
    dist.to_csv(OUT/"selected_class_distribution.csv", index=False)
    plt.figure(figsize=(9,5)); sns.barplot(data=dist, x="category", y="count")
    plt.xticks(rotation=20, ha="right"); plt.title("Selected Balanced Dataset")
    plt.tight_layout(); plt.savefig(FIG/"class_distribution.png", dpi=200); plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        data.text, data.label, test_size=.20, random_state=SEED, stratify=data.label)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
        ngram_range=(1,2), min_df=2, max_features=30000, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    pd.DataFrame([{"total_documents":len(data), "train_documents":len(X_train),
        "test_documents":len(X_test), "tfidf_features":Xtr.shape[1],
        "categories":" | ".join(chosen), "random_state":SEED}]).to_csv(OUT/"dataset_summary.csv", index=False)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED, max_depth=100, min_samples_leaf=2),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1),
        "Linear SVM": LinearSVC(C=1.0, random_state=SEED),
        "Naive Bayes": MultinomialNB(alpha=.5)
    }
    labels = sorted(data.label.unique())
    rows = []
    for name, model in models.items():
        t0=time.perf_counter(); model.fit(Xtr,y_train); train_time=time.perf_counter()-t0
        t0=time.perf_counter(); pred=model.predict(Xte); pred_time=time.perf_counter()-t0
        p,r,f,_=precision_recall_fscore_support(y_test,pred,average="macro",zero_division=0)
        rows.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),
            "Precision_macro":p,"Recall_macro":r,"F1_macro":f,
            "Train_seconds":train_time,"Predict_seconds":pred_time})
        slug=name.lower().replace(" ","_").replace("(","").replace(")","")
        pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T.to_csv(OUT/f"{slug}_classification_report.csv")
        cm=confusion_matrix(y_test,pred,labels=labels)
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/f"{slug}_confusion_matrix.csv")
        plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels)
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix – {name}")
        plt.xticks(rotation=30,ha="right"); plt.tight_layout()
        plt.savefig(FIG/f"{slug}_confusion_matrix.png",dpi=200); plt.close()

    metrics=pd.DataFrame(rows).sort_values("F1_macro",ascending=False)
    metrics.to_csv(OUT/"model_comparison.csv",index=False)
    print("\\nACTUAL HELD-OUT TEST RESULTS")
    print(metrics.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    melted=metrics.melt(id_vars="Model",value_vars=["Accuracy","Precision_macro","Recall_macro","F1_macro"],var_name="Metric",value_name="Score")
    plt.figure(figsize=(11,6)); sns.barplot(data=melted,x="Model",y="Score",hue="Metric")
    plt.ylim(0,1); plt.title("Classifier Performance Comparison")
    plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(FIG/"model_comparison.png",dpi=200); plt.close()

    # What-if experiment: only K changes; same split and same TF-IDF vectors.
    whatif=[]
    for k in [3,5,7]:
        model=KNeighborsClassifier(n_neighbors=k,metric="cosine",algorithm="brute",n_jobs=-1)
        model.fit(Xtr,y_train); pred=model.predict(Xte)
        p,r,f,_=precision_recall_fscore_support(y_test,pred,average="macro",zero_division=0)
        whatif.append({"K":k,"Accuracy":accuracy_score(y_test,pred),"Precision_macro":p,"Recall_macro":r,"F1_macro":f})
    wi=pd.DataFrame(whatif); wi.to_csv(OUT/"what_if_knn_k.csv",index=False)
    plt.figure(figsize=(8,5))
    for col in ["Accuracy","Precision_macro","Recall_macro","F1_macro"]:
        plt.plot(wi.K,wi[col],marker="o",label=col)
    plt.ylim(0,1); plt.xlabel("K neighbours"); plt.ylabel("Score")
    plt.title("What-if: KNN neighbour count"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG/"what_if_knn_k.png",dpi=200); plt.close()
    data[["text","label"]].to_csv(OUT/"selected_experiment_data.csv",index=False)
    (OUT/"experiment_summary.txt").write_text(
        f"Dataset: {files[0].name}\\nSelected categories: {chosen}\\nDocuments: {len(data)}\\n"
        f"Train: {len(X_train)}\\nTest: {len(X_test)}\\nTF-IDF features: {Xtr.shape[1]}\\n"
        f"Best model by macro F1: {metrics.iloc[0]['Model']} ({metrics.iloc[0]['F1_macro']:.4f})\\n",
        encoding="utf-8")
    print("\\nWHAT-IF RESULTS (KNN)")
    print(wi.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\\nSaved outputs/ and figures/. Use these actual files in your report.")

if __name__ == "__main__":
    main()
