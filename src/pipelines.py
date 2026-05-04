from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from src.tokenizer import tokenize

DEFAULT_SEED = 7524

_TFIDF_KWARGS = dict(
    ngram_range=(1, 2), min_df=1, max_df=0.95,
    sublinear_tf=True, norm='l2',
    analyzer='word', tokenizer=tokenize, token_pattern=None,
)


def build_pipe(clf, oversampler=None) -> ImbPipeline:
    steps = [('tfidf', TfidfVectorizer(**_TFIDF_KWARGS))]
    if oversampler is not None:
        steps.append(('oversample', oversampler))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def default_oversampler(seed: int = DEFAULT_SEED) -> RandomOverSampler:
    return RandomOverSampler(random_state=seed)


def default_classifiers(seed: int = DEFAULT_SEED) -> dict:
    return {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=2000, random_state=seed, n_jobs=1),
        'Naive Bayes': MultinomialNB(),
        'LinearSVC': LinearSVC(C=1.0, max_iter=2000, tol=1e-3, random_state=seed),
    }