import math

def keyword_hit(passages: list[dict], expected_keywords: list[str]) -> bool:
    #true if any expected keyword or prhase appears in the retrieved text
    if not expected_keywords:
        return False

    combined_text = ""
    for passage in passages:
        combined_text += passage["text"].lower() + " "

    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            return True

    return False

    
def page_hit(passages: list[dict], expected_page: int | None, tolerance: int = 3) -> bool | None:
    #true if any retrievd passages page is within tolerance pages of the page someone found the real answer on (page hint)
    if expected_page is None:
        return None

    for passage in passages:
        page_number = passage["metadata"]["pdf_page"]
        if abs(page_number - expected_page) <= tolerance:
            return True

    return False


def correct_manual_rate(passages: list[dict], expected_vehicle_id: str) -> float:
    #ratio of returned passages that came from the correct vehicles manual. 1 = every passage was, 0 = none were
    if not passages:
        return 0.0
    
    correct_count = 0
    for passage in passages:
        if passage["metadata"]["vehicle_id"] == expected_vehicle_id:
            correct_count += 1

    return correct_count / len(passages)


def top1_correct_manual(passages: list[dict], expected_vehicle_id: str) -> bool:
    if not passages:
        return False
    return passages[0]["metadata"]["vehicle_id"] == expected_vehicle_id


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += value

    return total / len(values)


def relevance(passage: dict, expected_vehicle_id: str | None, expected_keywords: list[str],
              page_hint: int | None) -> int:
    #scores how relevant the retrieved passage is (0 not relevant, 1 = right manual but not strong mathc, 2 = relevant)
    metadata = passage["metadata"]

    if expected_vehicle_id is not None and metadata["vehicle_id"] != expected_vehicle_id:
        return 0 

    text_lower = passage["text"].lower()
    keyword_matched = False
    for keyword in expected_keywords:
        if keyword.lower() in text_lower:
            keyword_matched = True
            break

    page_matched = False
    if page_hint is not None:
        page_matched = abs(metadata["pdf_page"] - page_hint) <= 3

    if keyword_matched or page_matched:
        return 2
    if expected_vehicle_id is not None:
        return 1
    return 0

def ncdcg(passages: list[dict], expected_vehicle_id: str | None, expected_keywords: list[str],
          page_hint: int | None, k: int = 5) -> float | None:
    #ranks the best passage landed. If a system puts the right answer 1st it scores higher than one that put it 5th even tho they both found it
    if expected_vehicle_id is None and not expected_keywords:
        return None

    grades = []
    for passage in passages[:k]:
        grade = relevance(passage, expected_vehicle_id, expected_keywords, page_hint)
        grades.append(grade)

    dcg = 0.0
    for rank, grade in enumerate(grades):
        dcg += grade / math.log2(rank + 2)

    ideal_grades = sorted(grades, reverse=True)
    idcg = 0.0
    for rank, grade in enumerate(ideal_grades):
        idcg += grade / math.log2(rank + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg