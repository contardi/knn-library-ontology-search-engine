# Library Ontology Search Engine

A semantic web-based library search engine that uses OWL ontologies and SPARQL queries to enable natural language book searches with age-appropriate filtering.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Flask](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![RDFlib](https://img.shields.io/badge/rdflib-7.0-orange.svg)](https://rdflib.readthedocs.io/)

## Features

### Semantic Search
- **Natural Language Search**: Search for books using everyday language (e.g., "fantasy books for 8 year olds")
- **Age-Appropriate Filtering**: Find books suitable for specific age groups (Early Childhood, Children, Middle Grade, Young Adult, Adult)
- **Genre Classification**: Browse books by fiction and non-fiction genres
- **OWL Ontology**: Well-structured ontology with class hierarchies, object properties, and data properties
- **SPARQL Queries**: Powerful semantic queries leveraging RDF triples

### Fallback Text Search
When semantic queries don't match structured parameters (genre, age, author), the system automatically falls back to a text-based search:

- **Multi-Field Search**: Searches across title, description, and author fields simultaneously
- **Stop Word Filtering**: Common words (the, a, for, with, etc.) are filtered out for better accuracy
- **Relevance Scoring**: Results are ranked by relevance with weighted scoring:
  - Title exact match: highest priority
  - Title contains term: high priority
  - Author match: medium priority
  - Description match: lower priority

### Smart Suggestions
- **Fuzzy Matching**: When no results are found, suggests similar titles and authors using string similarity algorithms
- **Match Highlighting**: Search terms are highlighted in results for easy identification

### API
- **REST API**: JSON endpoints for programmatic access

## Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose
- [Protege Desktop](https://protege.stanford.edu/) (optional, for viewing/editing the ontology)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/contardi/knn-library-ontology-search-engine.git
   cd knn-library-ontology-search-engine
   ```

2. **Build the Docker image**
   ```bash
   docker compose build
   ```

3. **Start the application**
   ```bash
   docker compose up
   ```

4. **Access the application**

   Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

## Project Structure

```
protege/
├── app/
│   ├── __init__.py
│   ├── app.py              # Flask application
│   ├── queries.py          # SPARQL queries and NL parser
│   ├── static/
│   │   └── style.css       # Stylesheet
│   └── templates/
│       ├── base.html       # Base template
│       ├── index.html      # Home page
│       └── results.html    # Search results page
├── data/
│   └── books_data.py       # Book data mock for populating ontology
├── images/                 # Screenshots and diagrams
├── ontology/
│   ├── library.owl         # Base ontology (OWL/XML)
│   ├── library.ttl         # Base ontology (Turtle)
│   ├── library_populated.owl   # Populated ontology (OWL/XML)
│   └── library_populated.ttl   # Populated ontology (Turtle)
├── scripts/
│   ├── create_ontology.py  # Creates base ontology structure
│   └── populate_ontology.py # Populates ontology with book data
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Usage

### Web Interface

The home page displays library statistics and provides:
- A search bar for natural language queries
- Browse options by genre
- Browse options by age group

### Example Queries

Try these natural language searches:

#### Semantic Queries (structured parameters)
| Query | Description |
|-------|-------------|
| `fantasy books for children` | Fantasy genre suitable for ages 6-8 |
| `science fiction for 12 year olds` | Sci-fi books for middle grade readers |
| `mystery books for adults` | Mystery novels for adult readers |
| `books by Stephen King` | Books by a specific author |
| `available thriller books` | Thriller books currently available |
| `young adult romance` | Romance books for teens (13-17) |

#### Fallback Text Search (when no structured match)
| Query | Description |
|-------|-------------|
| `Harry Potter` | Searches title, description, and author for "Harry Potter" |
| `dragons and magic` | Finds books mentioning dragons or magic |
| `Rowling` | Finds books by author name |
| `wizards` | Searches all text fields for the term |


## Ontology Details

### Class Hierarchy

The ontology defines the following main class hierarchies:

- **Book**
  - FictionBook (Novel, ShortStoryCollection, Poetry)
  - NonFictionBook (Biography, Educational, Reference, SelfHelp)

- **Genre**
  - FictionGenre (Mystery, Romance, ScienceFiction, Fantasy, Thriller, HistoricalFiction, Horror, Adventure)
  - NonFictionGenre (Science, History, Biography, Technology, Philosophy, Arts, Nature, Sports)

- **Person**
  - Author

- **AgeGroup** (EarlyChildhood, Children, MiddleGrade, YoungAdult, Adult)

### Object Properties

| Property | Domain | Range | Characteristics |
|----------|--------|-------|-----------------|
| hasAuthor | Book | Author | - |
| hasGenre | Book | Genre | - |
| forAgeGroup | Book | AgeGroup | - |
| hasFormat | Book | Format | - |
| hasStatus | Book | AvailabilityStatus | Functional |
| relatedTo | Book | Book | Symmetric |

### Data Properties

| Property | Domain | Range |
|----------|--------|-------|
| hasTitle | Book | xsd:string |
| hasISBN | Book | xsd:string |
| hasPublicationYear | Book | xsd:integer |
| hasPageCount | Book | xsd:integer |
| hasDescription | Book | xsd:string |
| hasAuthorName | Author | xsd:string |
| hasMinAge | AgeGroup | xsd:integer |
| hasMaxAge | AgeGroup | xsd:integer |

### Viewing in Protege

1. Download and install [Protege Desktop](https://protege.stanford.edu/)
2. Open `ontology/library_populated.owl` to view the full ontology with book data
3. Use the "Entities" tab to explore classes, properties, and individuals
4. Use the "DL Query" tab to run Description Logic queries

## Development

### Regenerating the Ontology

To regenerate the base ontology structure:

```bash
docker compose run --rm app python scripts/create_ontology.py
```

### Populating with Book Data

To populate the ontology with book data:

```bash
docker compose run --rm app python scripts/populate_ontology.py
```

### Adding New Books

1. Edit `data/books_data.py` to add new book entries
2. Run the populate script to regenerate the populated ontology
3. Restart the application to load the updated ontology

## Screenshots

The `images/` folder contains documentation screenshots:

| File | Description |
|------|-------------|
| [`01-class-hierarchy.png`](/images/01-class-hierarchy.png) | Ontology class hierarchy in Protege |
| [`02-object-properties.png`](/images/02-object-properties.png) | Object properties definition |
| [`03-data-properties.png`](/images/03-data-properties.png) | Data properties definition |
| [`04-individuals-by-class.png`](/images/04-individuals-by-class.png) | Individuals organized by class |
| [`05-hierarchy-graph.png`](/images/05-hierarchy-graph.png) | Visual graph of the ontology hierarchy |
| [`06-query-examples-01.png`](/images/06-query-examples-01.png) | SPARQL query examples (part 1) |
| [`06-query-examples-02.png`](/images/06-query-examples-02.png) | SPARQL query examples (part 2) |
| [`07-web-app-home-page.png`](/images/07-web-app-home-page.png) | Web Page Home Page Example |
| [`08-web-app-search-result-page.png`](/images/08-web-app-search-result-page.png) | Web Page Search Result Page |


## Demo

[WebM Format](/demo/demo-protege-web-app.webm)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was developed as part of the **MSc Artificial Intelligence** program at the **University of Essex**.