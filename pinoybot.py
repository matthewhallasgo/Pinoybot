import os
import pandas as pd
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def draw_box(lines, width=72):
    print("╔" + "═" * (width - 2) + "╗")
    for line in lines:
        print("║" + line.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")

def extract_features(tokens, index):
    word = tokens[index]
    word_lower = word.lower()
    
    features = {
        'word.lower': word_lower,
        'word.length': len(word),
        'word.is_capitalized': word[0].isupper() if word else False,
        'word.is_all_caps': word.isupper(),
        'word.is_all_lower': word.islower(),
        'word.contains_digit': any(char.isdigit() for char in word),
        'word.contains_punct': any(not char.isalnum() for char in word),
        'word.prefix_2': word_lower[:2] if len(word_lower) >= 2 else word_lower,
        'word.prefix_3': word_lower[:3] if len(word_lower) >= 3 else word_lower,
        'word.suffix_2': word_lower[-2:] if len(word_lower) >= 2 else word_lower,
        'word.suffix_3': word_lower[-3:] if len(word_lower) >= 3 else word_lower,
        'word.has_fil_prefix': any(word_lower.startswith(p) for p in ['nag', 'mag', 'pag', 'pa', 'na', 'um', 'in', 'ka']),
        'word.has_eng_suffix': any(word_lower.endswith(s) for s in ['ing', 'ed', 's', 'tion', 'ment', 'ity']),
    }
    
    if index > 0:
        prev_word = tokens[index - 1]
        features.update({
            'prev_word.lower': prev_word.lower(),
            'prev_word.is_capitalized': prev_word[0].isupper() if prev_word else False,
        })
    else:
        features['BOS'] = True

    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        features.update({
            'next_word.lower': next_word.lower(),
            'next_word.is_capitalized': next_word[0].isupper() if next_word else False,
        })
    else:
        features['EOS'] = True
        
    return features

def format_sid(sid):
    return int(sid) if float(sid).is_integer() else sid

if __name__ == "__main__":
    draw_box([
        "HELLO WORLD",
        "",
        "Welcome to Pinoybot!",
        "",
        "a CSINTSY project about the use of words in Filipino and English",
        "",
        "Made by: De Leon, Hallasgo, Lim, Rivera (MCO 23)"
    ])

    while True:
        user_start = input("\nPress Enter to start: ")
        if user_start == "":
            break
        else:
            print("[!] WARNING: Invalid input. Please just press Enter.")

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

    draw_box(["Training complete! Model is ready for use."])

    while True:
        show_metrics = input("\nDo you want to see the breakdown of precision? (Type 'yes' or press Enter to skip): ").strip().lower()
        if show_metrics == "":
            break
        elif show_metrics == "yes":
            print("\n" + report)
            break
        else:
            print("[!] WARNING: Invalid input. Please type 'yes' or press Enter to continue.")

    while True:
        print("\n")
        draw_box([
            "MAIN MENU",
            "1. Group Specific Finder (IDs 440-460)",
            "2. General Sentence Finder",
            "3. Exit Pinoybot"
        ])
        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            group_sids = [sid for sid in sentences_map.keys() if 440 <= sid <= 460]
            group_sids.sort()
            
            print("\nThe following sentence IDs are required for Group 23:")
            for idx, sid in enumerate(group_sids, 1):
                print(f"[{idx}] Sentence ID: {format_sid(sid)}")
            
            while True:
                user_run = input("\nPress Enter to execute the classifier on these sentences: ")
                if user_run == "":
                    break
                else:
                    print("[!] WARNING: Invalid input. Please just press Enter.")

            local_mapping = {}
            for idx, sid in enumerate(group_sids, 1):
                tokens = sentences_map[sid]['tokens']
                features_list = [extract_features(tokens, i) for i in range(len(tokens))]
                X_feat = vectorizer.transform(features_list)
                predictions = clf.predict(X_feat)
                
                local_mapping[idx] = {
                    'sid': sid,
                    'tokens': tokens,
                    'predictions': predictions
                }
            
            print("\nAnalysis Complete!")
            
            while True:
                view_choice = input(f"\nEnter a local sentence index to display (1-{len(group_sids)}) or 'back' to return: ").strip()
                if view_choice.lower() == 'back':
                    break
                try:
                    idx_val = int(view_choice)
                    if idx_val in local_mapping:
                        target = local_mapping[idx_val]
                        draw_box([f"Sentence {idx_val} (Original ID: {format_sid(target['sid'])})"])
                        print(f"{'WORD':<20} | TAG")
                        print("-" * 28)
                        for w, p in zip(target['tokens'], target['predictions']):
                            print(f"{w:<20} | {p}")
                        print("-" * 28)
                    else:
                        print(f"[!] WARNING: Out of range. Please enter a number between 1 and {len(group_sids)}.")
                except ValueError:
                    print("[!] WARNING: Invalid input. Please enter a valid number.")

        elif choice == "2":
            all_sids = list(sentences_map.keys())
            
            while True:
                search_input = input("\nEnter a sentence ID to search, 'random' for a random choice, or 'done' to stop: ").strip()
                
                if search_input.lower() == 'done':
                    break
                
                if search_input.lower() == 'random':
                    target_sid = random.choice(all_sids)
                else:
                    try:
                        target_sid = float(search_input)
                        if target_sid not in sentences_map:
                            print("[!] WARNING: Sentence ID not found in the dataset. Try another one.")
                            continue
                    except ValueError:
                        print("[!] WARNING: Invalid search parameter. Enter a valid number, 'random', or 'done'.")
                        continue
                
                tokens = sentences_map[target_sid]['tokens']
                features_list = [extract_features(tokens, i) for i in range(len(tokens))]
                X_feat = vectorizer.transform(features_list)
                predictions = clf.predict(X_feat)
                
                print("\n")
                draw_box([f"Sentence ID: {format_sid(target_sid)}"])
                print(f"{'WORD':<20} | TAG")
                print("-" * 28)
                for w, p in zip(tokens, predictions):
                    print(f"{w:<20} | {p}")
                print("-" * 28)

        elif choice == "3":
            draw_box(["Thank you for using Pinoybot! Goodbye."])
            sys.exit()
        else:
            print("[!] WARNING: Invalid choice. Please pick option 1, 2, or 3.")