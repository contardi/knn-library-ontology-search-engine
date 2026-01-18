"""
SPARQL query functions and natural language parser for the Library Search Engine.
"""

import re
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

LIB = Namespace("http://example.org/library#")

# Genre mappings for natural language parsing
FICTION_GENRES = [
    "mystery", "romance", "science fiction", "sci-fi", "fantasy",
    "thriller", "historical fiction", "horror", "adventure"
]

NONFICTION_GENRES = [
    "science", "history", "biography", "technology", "philosophy", "arts", "nature", "sports"
]

ALL_GENRES = FICTION_GENRES + NONFICTION_GENRES

# Genre name normalization
GENRE_NORMALIZATION = {
    "sci-fi": "ScienceFiction",
    "science fiction": "ScienceFiction",
    "historical fiction": "HistoricalFiction",
    "mystery": "Mystery",
    "romance": "Romance",
    "fantasy": "Fantasy",
    "thriller": "Thriller",
    "horror": "Horror",
    "adventure": "Adventure",
    "science": "Science",
    "history": "History",
    "biography": "BiographyGenre",
    "technology": "Technology",
    "philosophy": "Philosophy",
    "arts": "Arts",
    "nature": "Nature",
    "sports": "Sports",
}

# Age group mappings
AGE_GROUPS = {
    "early childhood": ("EarlyChildhood", 0, 5),
    "toddler": ("EarlyChildhood", 0, 5),
    "preschool": ("EarlyChildhood", 0, 5),
    "children": ("Children", 6, 8),
    "kids": ("Children", 6, 8),
    "middle grade": ("MiddleGrade", 9, 12),
    "tween": ("MiddleGrade", 9, 12),
    "young adult": ("YoungAdult", 13, 17),
    "ya": ("YoungAdult", 13, 17),
    "teen": ("YoungAdult", 13, 17),
    "adult": ("Adult", 18, 99),
}


def parse_natural_language_query(query: str) -> dict:
    """
    Parse a natural language query to extract search parameters.

    Examples:
    - "Fantasy books for children with 8 years old"
    - "Science fiction for adults"
    - "Mystery books"
    - "Books by Stephen King"
    """
    query_lower = query.lower().strip()

    result = {
        "genre": None,
        "genre_instance": None,
        "age": None,
        "age_group": None,
        "category": None,  # fiction or non-fiction
        "author": None,
        "title_keywords": [],
        "status": None,
    }

    # Extract age (e.g., "8 years old", "for 8 year olds", "age 8")
    age_patterns = [
        r"(\d+)\s*years?\s*old",
        r"for\s*(\d+)\s*year",
        r"age\s*(\d+)",
        r"(\d+)\s*yo\b",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            result["age"] = int(match.group(1))
            break

    # Extract genre
    for genre_key, genre_value in GENRE_NORMALIZATION.items():
        if genre_key in query_lower:
            result["genre"] = genre_value
            result["genre_instance"] = f"{genre_value}Instance"
            # Determine category based on genre
            if genre_key in FICTION_GENRES:
                result["category"] = "fiction"
            else:
                result["category"] = "non-fiction"
            break

    # Extract age group by name
    for age_key, (age_group, min_age, max_age) in AGE_GROUPS.items():
        if age_key in query_lower:
            result["age_group"] = age_group
            # If no specific age, use midpoint of range
            if result["age"] is None:
                result["age"] = (min_age + max_age) // 2
            break

    # If we have an age but no age group, determine the age group
    if result["age"] is not None and result["age_group"] is None:
        age = result["age"]
        if age <= 5:
            result["age_group"] = "EarlyChildhood"
        elif age <= 8:
            result["age_group"] = "Children"
        elif age <= 12:
            result["age_group"] = "MiddleGrade"
        elif age <= 17:
            result["age_group"] = "YoungAdult"
        else:
            result["age_group"] = "Adult"

    # Extract author (e.g., "by Stephen King", "author Stephen King")
    author_patterns = [
        r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"author\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    for pattern in author_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            result["author"] = match.group(1)
            break

    # Extract availability status
    if "available" in query_lower:
        result["status"] = "Available"
    elif "checked out" in query_lower or "borrowed" in query_lower:
        result["status"] = "CheckedOut"

    # Extract category if not already set
    if result["category"] is None:
        if "fiction" in query_lower and "non-fiction" not in query_lower and "nonfiction" not in query_lower:
            result["category"] = "fiction"
        elif "non-fiction" in query_lower or "nonfiction" in query_lower:
            result["category"] = "non-fiction"

    # If no specific parameters found, use entire query as title search
    if not any([result["genre"], result["age_group"], result["author"],
                result["category"], result["status"]]):
        result["title_keywords"] = query.strip()

    return result


def build_sparql_query(params: dict) -> str:
    """Build a SPARQL query from parsed parameters."""

    # Base query parts
    select_vars = ["?book", "?title", "?authorName", "?genreLabel", "?ageGroupLabel", "?year", "?description", "?status"]

    where_clauses = [
        "?book rdf:type ?bookType .",
        "?bookType rdfs:subClassOf* lib:Book .",
        "?book lib:hasTitle ?title .",
    ]

    optional_clauses = [
        "OPTIONAL { ?book lib:hasAuthor ?author . ?author lib:hasAuthorName ?authorName . }",
        "OPTIONAL { ?book lib:hasGenre ?genre . ?genre rdfs:label ?genreLabel . }",
        "OPTIONAL { ?book lib:forAgeGroup ?ageGroup . ?ageGroup lib:hasAgeGroupName ?ageGroupLabel . }",
        "OPTIONAL { ?book lib:hasPublicationYear ?year . }",
        "OPTIONAL { ?book lib:hasDescription ?description . }",
        "OPTIONAL { ?book lib:hasStatus ?statusObj . ?statusObj rdfs:label ?status . }",
    ]

    filters = []

    # Add genre filter
    if params.get("genre_instance"):
        where_clauses.append(f"?book lib:hasGenre lib:{params['genre_instance']} .")

    # Add age group filter
    if params.get("age_group"):
        where_clauses.append(f"?book lib:forAgeGroup lib:{params['age_group']} .")
    elif params.get("age") is not None:
        # Filter by age range
        age = params["age"]
        where_clauses.extend([
            "?book lib:forAgeGroup ?ageGroupFilter .",
            "?ageGroupFilter lib:hasMinAge ?minAge .",
            "?ageGroupFilter lib:hasMaxAge ?maxAge .",
        ])
        filters.append(f"FILTER(?minAge <= {age} && ?maxAge >= {age})")

    # Add author filter
    if params.get("author"):
        author = params["author"].replace("'", "\\'")
        where_clauses.append("?book lib:hasAuthor ?authorFilter .")
        where_clauses.append("?authorFilter lib:hasAuthorName ?authorFilterName .")
        filters.append(f"FILTER(CONTAINS(LCASE(?authorFilterName), LCASE('{author}')))")

    # Add category filter (fiction/non-fiction)
    if params.get("category") == "fiction":
        where_clauses.append("?book rdf:type/rdfs:subClassOf* lib:FictionBook .")
    elif params.get("category") == "non-fiction":
        where_clauses.append("?book rdf:type/rdfs:subClassOf* lib:NonFictionBook .")

    # Add status filter
    if params.get("status"):
        where_clauses.append(f"?book lib:hasStatus lib:{params['status']} .")

    # Add title keyword filter
    if params.get("title_keywords"):
        title_search = params["title_keywords"].replace("'", "\\'")
        filters.append(f"FILTER(CONTAINS(LCASE(?title), LCASE('{title_search}')))")

    # Build the query
    query = f"""
    PREFIX lib: <http://example.org/library#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT DISTINCT {' '.join(select_vars)}
    WHERE {{
        {' '.join(where_clauses)}
        {' '.join(optional_clauses)}
        {' '.join(filters)}
    }}
    ORDER BY ?title
    """

    return query


def search_books(g: Graph, query_text: str) -> list:
    """
    Search books using natural language query.
    Returns a list of book dictionaries.
    """
    params = parse_natural_language_query(query_text)
    sparql_query = build_sparql_query(params)

    results = []
    for row in g.query(sparql_query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)

    return results, params


def get_all_books(g: Graph, limit: int = 50) -> list:
    """Get all books with their details."""

    query = f"""
    PREFIX lib: <http://example.org/library#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?book ?title ?authorName ?genreLabel ?ageGroupLabel ?year ?description ?status
    WHERE {{
        ?book rdf:type ?bookType .
        ?bookType rdfs:subClassOf* lib:Book .
        ?book lib:hasTitle ?title .
        OPTIONAL {{ ?book lib:hasAuthor ?author . ?author lib:hasAuthorName ?authorName . }}
        OPTIONAL {{ ?book lib:hasGenre ?genre . ?genre rdfs:label ?genreLabel . }}
        OPTIONAL {{ ?book lib:forAgeGroup ?ageGroup . ?ageGroup lib:hasAgeGroupName ?ageGroupLabel . }}
        OPTIONAL {{ ?book lib:hasPublicationYear ?year . }}
        OPTIONAL {{ ?book lib:hasDescription ?description . }}
        OPTIONAL {{ ?book lib:hasStatus ?statusObj . ?statusObj rdfs:label ?status . }}
    }}
    ORDER BY ?title
    LIMIT {limit}
    """

    results = []
    for row in g.query(query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)

    return results


def get_books_by_genre(g: Graph, genre: str) -> list:
    """Get all books of a specific genre."""
    params = {"genre_instance": f"{genre}Instance"}
    sparql_query = build_sparql_query(params)
    results = []
    for row in g.query(sparql_query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)
    return results


def get_books_by_age_group(g: Graph, age_group: str) -> list:
    """Get all books for a specific age group."""
    params = {"age_group": age_group}
    sparql_query = build_sparql_query(params)
    results = []
    for row in g.query(sparql_query):
        book = {
            "uri": str(row.book),
            "title": str(row.title) if row.title else "",
            "author": str(row.authorName) if row.authorName else "Unknown",
            "genre": str(row.genreLabel) if row.genreLabel else "",
            "age_group": str(row.ageGroupLabel) if row.ageGroupLabel else "",
            "year": int(row.year) if row.year else None,
            "description": str(row.description) if row.description else "",
            "status": str(row.status) if row.status else "Unknown",
        }
        results.append(book)
    return results


def get_statistics(g: Graph) -> dict:
    """Get library statistics."""

    # Total books
    books_query = """
    PREFIX lib: <http://example.org/library#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT (COUNT(DISTINCT ?book) as ?bookCount)
    WHERE {
        ?book lib:hasTitle ?title .
    }
    """

    # Total authors
    authors_query = """
    PREFIX lib: <http://example.org/library#>
    SELECT (COUNT(DISTINCT ?author) as ?authorCount)
    WHERE {
        ?author lib:hasAuthorName ?name .
    }
    """

    # Available books
    available_query = """
    PREFIX lib: <http://example.org/library#>
    SELECT (COUNT(DISTINCT ?book) as ?availableCount)
    WHERE {
        ?book lib:hasStatus lib:Available .
    }
    """

    stats = {
        "total_books": 0,
        "total_authors": 0,
        "available_books": 0,
    }

    for row in g.query(books_query):
        stats["total_books"] = int(row.bookCount)

    for row in g.query(authors_query):
        stats["total_authors"] = int(row.authorCount)

    for row in g.query(available_query):
        stats["available_books"] = int(row.availableCount)

    return stats
