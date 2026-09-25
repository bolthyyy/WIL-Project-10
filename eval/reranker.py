from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sort_by_score(passages: list[dict], scores) -> list[dict]:
    #pairs each passage with its score then sorts highest to lowest.
    scored_passages = []
    for passage, score in zip(passages, scores):
        scored_passages.append((score, passage))

    scored_passages.sort(key=lambda pair: pair[0], reverse=True)

    sorted_passages = []
    for score, passage in scored_passages:
        sorted_passages.append(passage)

    return sorted_passages


def rerank(question: str, passages: list[dict], top_k: int | None = None) -> list[dict]:
    if not passages:
        return passages

    texts = []
    for passage in passages:
        texts.append(passage["text"])

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [question])
    query_vector = matrix[-1]
    passage_vectors = matrix[:-1]

    scores = cosine_similarity(query_vector, passage_vectors)[0]
    reranked = sort_by_score(passages, scores)

    if top_k:
        reranked = reranked[:top_k]

    return reranked

def cross_encoder_rerank( question: str, passages: list[dict], top_k: int | None = None,
                         model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[dict]:
    #use just for locally cos huggingface.co is unreachable here
    if not passages:
        return passages

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)

    pairs = []
    for passage in passages:
        pairs.append((question, passage["text"]))

    scores = model.predict(pairs)

    reranked = sort_by_score(passages, scores)

    if top_k:
        reranked = reranked[:top_k]

    return reranked