from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# Default TF-IDF config — matches binary experiment settings
DEFAULT_TFIDF = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    norm='l2',
)

# Default oversampler — RandomOverSampler won all per-game comparisons in binary exp
DEFAULT_SEED = 7524


def build_pipe(clf, oversampler=None, tfidf_cfg: dict | None = None) -> ImbPipeline:
    """
    Build TF-IDF → oversample → clf pipeline.
    oversampler=None skips oversampling step (useful for anomaly detection).
    """
    tfidf_cfg = tfidf_cfg or DEFAULT_TFIDF
    steps = [('tfidf', TfidfVectorizer(**tfidf_cfg))]
    if oversampler is not None:
        steps.append(('oversample', oversampler))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def default_classifiers(seed: int = DEFAULT_SEED) -> dict:
    """Return dict of standard classifiers used across all experiments."""
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed, n_jobs=1
        ),
        'Naive Bayes': MultinomialNB(),
        'LinearSVC': LinearSVC(C=1.0, max_iter=2000, tol=1e-3, random_state=seed),
    }


def default_oversampler(seed: int = DEFAULT_SEED) -> RandomOverSampler:
    return RandomOverSampler(random_state=seed)
