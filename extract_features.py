from farasa.segmenter import FarasaSegmenter
import os
def extract_morphological_features(sentence):
    # Initialize Farasa tools
    segmenter = FarasaSegmenter()
    # Segment the sentence into words
    segmented_words = segmenter.segment(sentence)
    # Return the results
    return {
        'segmented_words': segmented_words,
    }

def extract_orthographic_features(sentence):
    # Initialize a dictionary to store orthographic features
    orthographic_features = {}

    # Extract word length for each word in the sentence
    words = sentence.split()
    orthographic_features['word_lengths'] = [len(word) for word in words]

    # Check if each word is capitalized
    orthographic_features['is_defined'] = [word[:3] == '+ال' for word in words]

    # Check for repeated characters in each word
    orthographic_features['has_repeated_characters'] = [any(word.count(char) > 1 for char in word) for word in words]

    return orthographic_features


def extract_contextual_features(sentence):
    # Initialize a dictionary to store contextual features
    contextual_features = {}

    # Tokenize the sentence into words
    words = sentence.split()

    # Extract contextual features for each word
    for i, word in enumerate(words):
        # Features for the current word
        current_word_features = {}

        # Previous word (if exists)
        if i > 0:
            current_word_features['previous_word'] = words[i - 1]
        else:
            current_word_features['previous_word'] = None

        # Next word (if exists)
        if i < len(words) - 1:
            current_word_features['next_word'] = words[i + 1]
        else:
            current_word_features['next_word'] = None

        # Add the features to the dictionary
        contextual_features[word] = current_word_features

    return contextual_features



import spacy

def extract_syntactic_features(sentence):
    # Load the Arabic language model for spaCy
    nlp = spacy.load("xx_ent_wiki_sm")  # Make sure to download the model using: python -m spacy download xx_ent_wiki_sm

    # Process the sentence with spaCy
    doc = nlp(sentence)

    # Extract syntactic features for each word
    syntactic_features = {}

    for token in doc:
        # Features for the current word
        current_word_features = {
            'word': token.text,
            'lemma': token.lemma_,
            'pos_tag': token.pos_,
            'dependency_relation': token.dep_,
            'head_word': token.head.text,
            'is_stopword': token.is_stop
        }

        # Add the features to the dictionary
        syntactic_features[token.i] = current_word_features

    return syntactic_features

# Example usage:
arabic_sentence = "ذهب محمد إلى المدرسة"
syntactic_features = extract_syntactic_features(arabic_sentence)

# Display the results for each word
for word_id, features in syntactic_features.items():
    print(f"Word {word_id + 1}:")
    print(f"  - Word: {features['word']}")
    print(f"  - Lemma: {features['lemma']}")
    print(f"  - POS Tag: {features['pos_tag']}")
    print(f"  - Dependency Relation: {features['dependency_relation']}")
    print(f"  - Head Word: {features['head_word']}")
    print(f"  - Is Stopword: {features['is_stopword']}")
    print("-------------------")
