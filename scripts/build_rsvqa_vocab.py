import json
from collections import Counter
import os

with open('datasets/rsvqa/LR_split_train_questions.json') as f:
    questions = json.load(f)["questions"]

with open('datasets/rsvqa/LR_split_train_answers.json') as f:
    answers = json.load(f)["answers"]

q_words = Counter()
for q in questions:
    if not q.get("active", False) or "question" not in q:
        continue
    text = q["question"].lower().replace('?', '').replace(',', '').split()
    q_words.update(text)

vocab = {word: i+2 for i, (word, count) in enumerate(q_words.most_common(1000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

ans_words = Counter()
for a in answers:
    if not a.get("active", False) or "answer" not in a:
        continue
    ans_words[str(a["answer"])] += 1

ans_vocab = {word: i for i, (word, count) in enumerate(ans_words.most_common())}

os.makedirs('datasets/processed/rsvqa', exist_ok=True)
with open('datasets/processed/rsvqa/q_vocab.json', 'w') as f:
    json.dump(vocab, f)
with open('datasets/processed/rsvqa/a_vocab.json', 'w') as f:
    json.dump(ans_vocab, f)

print(f"Question vocab size: {len(vocab)}")
print(f"Answer vocab size: {len(ans_vocab)}")
