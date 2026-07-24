import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.metrics import classification_report
from pinoybot import extract_features, draw_box

def train_and_save():
    draw_box(["Loading data..."])
    file_path = 'Group 23 MCO2 dataset_annotation (updated).xlsx'
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

    draw_box(["Splitting dataset (70-15-15)..."])
    sids = list(sentences_map.keys())
    sids_train, sids_temp = train_test_split(sids, test_size=0.3, random_state=67)
    sids_val, sids_test = train_test_split(sids_temp, test_size=0.5, random_state=67)

    draw_box(["Extracting features..."])
    def build_xy(sid_list):
        X, y = [], []
        for sid in sid_list:
            data = sentences_map[sid]
            for i in range(len(data['tokens'])):
                X.append(extract_features(data['tokens'], i))
                y.append(data['labels'][i])
        return X, y

    X_train_dicts, y_train = build_xy(sids_train)
    X_val_dicts, y_val = build_xy(sids_val)
    X_test_dicts, y_test = build_xy(sids_test)

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_val = vectorizer.transform(X_val_dicts)
    X_test = vectorizer.transform(X_test_dicts)

    draw_box(["Training Extra Trees Classifier..."])
    
    custom_weights = {
        'FIL': 1.0,
        'ENG': 2.2,    
        'OTH': 2.0,
        'CS': 60.0     
    }

    clf = ExtraTreesClassifier(
        n_estimators=500,   
        class_weight=custom_weights,
        min_samples_leaf=1,
        min_samples_split=2,
        criterion='entropy',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    def predict_with_threshold(X, model, threshhold=0.13):
        y_probs = model.predict_proba(X)
        classes = list(model.classes_)
            
        cs_index = classes.index('CS')
        y_pred_custom = []
        
        for prob_vector in y_probs:
            if prob_vector[cs_index] >= threshhold:
                y_pred_custom.append('CS')
            else:
                y_pred_custom.append(classes[np.argmax(prob_vector)])
        return y_pred_custom

    draw_box(["Evaluating Model on Validation Set..."])
    y_pred_val = predict_with_threshold(X_val, clf, 0.13)
    report_val = classification_report(y_val, y_pred_val, zero_division=0)
    print("\n" + report_val + "\n")

    draw_box(["Evaluating Model on Test Set..."])
    y_pred_test = predict_with_threshold(X_test, clf, 0.13)
    report_test = classification_report(y_test, y_pred_test, zero_division=0)
    print("\n" + report_test + "\n")

    draw_box(["Saving models and maps to disk..."])
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(clf, 'model.pkl')
    joblib.dump(sentences_map, 'sentences_map.pkl')
    draw_box(["Training complete! All files generated successfully.",
              "",
              "Files saved. Please run pinoybot.py to use the trained model."])

if __name__ == "__main__":
    train_and_save()
