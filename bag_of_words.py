from collections import Counter

# Implements the bag of words algorith.
#
# Based on an exercise from Udacity.com
################################################################
def bag_of_words(text):
    return Counter(text.split())

test_text = 'the quick brown fox jumps over the lazy dog'

print(bag_of_words(test_text))
