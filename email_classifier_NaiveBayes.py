from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#cats = ['rec.sport.baseball', 'rec.sport.hockey']
#cats = ['comp.sys.ibm.pc.hardware','rec.sport.hockey']
#cats = ['talk.politics.misc', 'talk.religion.misc', 'talk.politics.mideast', 'talk.politics.guns']
cats = ['talk.religion.misc', 'soc.religion.christian']
emails = fetch_20newsgroups(categories = cats)

print(emails.target_names)
#print(emails.data[5])
print(emails.target[5])

train_emails = fetch_20newsgroups(categories = cats, subset = 'train', shuffle = True, random_state = 108)

test_emails = fetch_20newsgroups(categories = cats, subset = 'test', shuffle = True, random_state = 108)

counter = CountVectorizer()
counter.fit(train_emails.data + test_emails.data)

train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

acc = classifier.score(test_counts, test_emails.target)
print(acc)