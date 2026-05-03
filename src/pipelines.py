from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# word + char TF-IDF union — char captures obfuscations and multilingual toxicity
DEFAULT_TFIDF = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True, norm='l2')),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1, max_df=0.95, sublinear_tf=True, norm='l2')),
])

DEFAULT_SEED = 7524


def default_oversampler(seed: int = DEFAULT_SEED) -> RandomOverSampler:
    return RandomOverSampler(random_state=seed)


def build_pipe(clf, oversampler=None, tfidf=None) -> ImbPipeline:
    # clone tfidf so each pipe owns independent state — prevents shared vocab corruption
    steps = [('tfidf', clone(tfidf if tfidf is not None else DEFAULT_TFIDF))]
    if oversampler is not None:
        steps.append(('oversample', oversampler))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def default_classifiers(seed: int = DEFAULT_SEED) -> dict:
    return {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=1),
        'Naive Bayes': MultinomialNB(),
        'LinearSVC': LinearSVC(C=1.0, max_iter=2000, tol=1e-3, random_state=seed),
    }
