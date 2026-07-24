import os
import sys
import random
import joblib

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
        'word.has_fil_prefix': any(word_lower.startswith(p) for p in ['nag', 'mag', 'pag', 'ma', 'pa', 'na', 'um', 'in', 'ka', 'ipa', 'ipag', 'ipang', 'pala', 'paki', 'mala', 'pang', 'nang', 'mang', 'pinag']),
        'word.has_fil_suffix': any(word_lower.endswith(p) for p in ['han', 'hin', 'syon', 'ilyo']),
        'word.has_eng_prefix': any(word_lower.startswith(s) for s in ['un', 're', 'dis', 'over', 'under', 'anti', 'post', 'pre', 'non', 'sub']),
        'word.has_eng_suffix': any(word_lower.endswith(s) for s in ['ing', 'ed', 's', 'tion', 'ment', 'ity', 'ness', 'able', 'ful', 'less', 'al']),
        'word.is_single_repeated_char': is_single_repeated_char(word_lower)
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

def is_single_repeated_char(word_lower):
   return len(word_lower) > 0 and len(set(word_lower)) == 1

def tag_language(tokens):
    if not os.path.exists('vectorizer.pkl') or not os.path.exists('model.pkl'):
        print("[!] ERROR: Model files missing. Please run train.py first.")
        sys.exit()
        
    vectorizer = joblib.load('vectorizer.pkl')
    clf = joblib.load('model.pkl')
    
    features_list = [extract_features(tokens, i) for i in range(len(tokens))]
    X_feat = vectorizer.transform(features_list)
    predictions = clf.predict(X_feat)
    
    return list(predictions)

def format_sid(sid):
    return int(sid) if float(sid).is_integer() else sid

if __name__ == "__main__":
    draw_box([
        "HELLO WORLD",
        "",
        "Welcome to Pinoybot!",
        "",
        "A CSINTSY project about the use of words in Filipino and English",
        "",
        "Made by: De Leon, Hallasgo, Lim, Rivera (MCO2 23)"
    ])

    while True:
        user_start = input("\nPress Enter to start. ")
        if user_start == "":
            break
        else:
            print("[!] WARNING: Invalid input. Please just press Enter.")

    if not os.path.exists('sentences_map.pkl'):
        print("[!] ERROR: Saved sentences map data missing. Run train.py first.")
        sys.exit()
        
    sentences_map = joblib.load('sentences_map.pkl')

    while True:
        print("\n")
        draw_box([
            "MAIN MENU",
            "1. Group Specific Finder (IDs 440-460)",
            "2. General Sentence Finder",
            "3. Custom Sentence Input",
            "4. Exit Pinoybot"
        ])
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            group_sids = [sid for sid in sentences_map.keys() if 440 <= sid <= 460]
            group_sids.sort()
            
            print("\nThe following sentence IDs are required for Group 23:")
            for idx, sid in enumerate(group_sids, 1):
                print(f"[{idx}] Sentence ID: {format_sid(sid)}")
            
            while True:
                user_run = input("\nPress Enter to execute the classifier on these sentences.")
                if user_run == "":
                    break
                else:
                    print("[!] WARNING: Invalid input. Please just press Enter.")

            local_mapping = {}
            for idx, sid in enumerate(group_sids, 1):
                tokens = sentences_map[sid]['tokens']
                predictions = tag_language(tokens)
                
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
                predictions = tag_language(tokens)
                
                print("\n")
                draw_box([f"Sentence ID: {format_sid(target_sid)}"])
                print(f"{'WORD':<20} | TAG")
                print("-" * 28)
                for w, p in zip(tokens, predictions):
                    print(f"{w:<20} | {p}")
                print("-" * 28)

        elif choice == "3":
            while True:
                custom_input = input("\nEnter a sentence to classify: ").strip()
                if custom_input:
                    tokens = custom_input.split()
                    predictions = tag_language(tokens)
                    print("\n")
                    draw_box(["Custom Input Analysis"])
                    print(f"{'WORD':<20} | TAG")
                    print("-" * 28)
                    for w, p in zip(tokens, predictions):
                        print(f"{w:<20} | {p}")
                    print("-" * 28)
                
                while True:
                    another = input("\nDo you want to input another sentence? (Type 'yes' to continue or press Enter to return to main menu): ").strip().lower()
                    if another == "yes" or another == "":
                        break
                    else:
                        print("[!] WARNING: Invalid input. Please type 'yes' or press Enter.")
                        
                if another != "yes":
                    break

        elif choice == "4":
            draw_box(["Thank you for using Pinoybot! Goodbye."])
            sys.exit()
        else:
            print("[!] WARNING: Invalid choice. Please pick option 1, 2, 3, or 4.")