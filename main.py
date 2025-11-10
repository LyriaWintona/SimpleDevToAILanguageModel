import numpy as np

rage = """Artificial Intelligence is stupid. It cant learn a thing, and has no idea what is going to come next. AI should deconstruct itself before its too late"""

# This string just has all of its vocab in it

words = rage.lower().split() # makes it all lower case, and splits it into words?

vocab = list(set(words)) # self explanatory, just puts all of words in a list
vocab_size = len(vocab) # self explanatory, just takes the amount of words, and sets it as an integer

print(f"Vocabulary: {vocab}") # shows us all words in vocabulary
print(f"Vocabulary size: {vocab_size}")  # Shows us the size of its vocabulary

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Convert the words in the rage to indices
rage_indices = [word_to_idx[word] for word in words]

# Initialize bigram counts matrix
bigram_counts = np.zeros((vocab_size, vocab_size))

# Count occurrences of each bigram in the corpus
for i in range(len(rage_indices) - 1):
    current_word = rage_indices[i]
    next_word = rage_indices[i + 1]
    bigram_counts[current_word, next_word] += 1

# Apply Laplace smoothing by adding 1 to all bigram counts
bigram_counts += 0.01

# Normalize the counts to get probabilities
bigram_probabilities = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)

print("Bigram probabilities matrix: ", bigram_probabilities)

def predict_next_word(current_word, bigram_probabilities):
    word_idx = word_to_idx[current_word]
    next_word_probs = bigram_probabilities[word_idx]
    next_word_idx = np.random.choice(range(vocab_size), p=next_word_probs)
    return idx_to_word[next_word_idx]

# Test the model with a word
current_word = "ai"
next_word = predict_next_word(current_word, bigram_probabilities)
print(f"Given '{current_word}', the model predicts '{next_word}'.")

def generate_sentence(start_word, bigram_probabilities, length=5):
    sentence = [start_word]
    current_word = start_word

    for _ in range(length):
        next_word = predict_next_word(current_word, bigram_probabilities)
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)

# Generate a sentence starting with "artificial"
generated_sentence = generate_sentence("artificial", bigram_probabilities, length=10)
print(f"Generated sentence: {generated_sentence}")

