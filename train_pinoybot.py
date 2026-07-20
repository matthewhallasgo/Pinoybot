import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from pinoybot import extract_features, draw_box

def train_and_save():
    draw_box(["Loading data..."])
    file_path = 'Group 23 MCO2 dataset_annotation.xlsx'
    df = pd.read_excel(file_path, sheet_name='FINAL ANNOTATION')
    
    df = df.dropna(subset=['word', 'answer', 'sentence_id'])
    df['word'] = df['word'].astype(str)
    df['answer'] = df['answer'].astype(str).str.strip().str.upper()
    df['sentence_id'] = df['sentence_id'].astype(float)
    
    valid_labels = ['FIL', 'ENG', 'CS', 'OTH']
    df = df[df['answer'].isin(valid_labels)]

    sentences_map = {}
    for sid, group in df.groupby('sentence_id', sort=False):
        sentences_map[sid] = {
            'tokens': group['word'].tolist(),
            'labels': group['answer'].tolist()
        }

    draw_box(["Extracting features..."])
    X_all, y_all = [], []
    for sid, data in sentences_map.items():
        for i in range(len(data['tokens'])):
            X_all.append(extract_features(data['tokens'], i))
            y_all.append(data['labels'][i])

    draw_box(["Splitting dataset (70-15-15)..."])
    X_train_dicts, X_temp_dicts, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    X_val_dicts, X_test_dicts, y_val, y_test = train_test_split(
        X_temp_dicts, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_test = vectorizer.transform(X_test_dicts)

    draw_box(["Training Decision Tree Classifier..."])
    clf = DecisionTreeClassifier(max_depth=30, min_samples_split=5, random_state=42)
    clf.fit(X_train, y_train)

    draw_box(["Evaluating Model on Test Set..."])
    y_pred_test = clf.predict(X_test)
    report = classification_report(y_test, y_pred_test)
    
    print("\n" + report + "\n")

    draw_box(["Saving models and maps to disk..."])
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(clf, 'model.pkl')
    joblib.dump(sentences_map, 'sentences_map.pkl')
    draw_box(["Training complete! All files generated successfully.",
              "",
              "Files saved. Please run pinoybot.py to use the trained model."])

if __name__ == "__main__":
    train_and_save()