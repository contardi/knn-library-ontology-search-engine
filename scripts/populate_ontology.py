#!/usr/bin/env python3
"""
Populate the Library Ontology with book and author instances.
This script loads the base ontology and adds all books and authors from the mock data.
"""

import os
import sys

# Add parent directory to path to import data module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from data.books_data import get_all_authors, get_all_books

# Define namespaces
LIB = Namespace("http://example.org/library#")


def populate_ontology(g: Graph) -> Graph:
    """Add authors and books to the ontology graph."""

    # Bind namespace
    g.bind("lib", LIB)

    authors = get_all_authors()
    books = get_all_books()

    print(f"Adding {len(authors)} authors...")

    # Add all authors
    for author in authors:
        author_uri = LIB[author["id"]]
        g.add((author_uri, RDF.type, LIB["Author"]))
        g.add((author_uri, RDF.type, OWL.NamedIndividual))
        g.add((author_uri, LIB["hasAuthorName"], Literal(author["name"])))
        g.add((author_uri, LIB["hasBirthYear"], Literal(author["birth_year"], datatype=XSD.integer)))
        g.add((author_uri, LIB["hasNationality"], Literal(author["nationality"])))
        g.add((author_uri, RDFS.label, Literal(author["name"])))

    print(f"Adding {len(books)} books...")

    # Add all books
    for book in books:
        book_uri = LIB[book["id"]]

        # Add book type (FictionBook, Novel, NonFictionBook, etc.)
        book_type = book.get("book_type", "Book")
        g.add((book_uri, RDF.type, LIB[book_type]))
        g.add((book_uri, RDF.type, OWL.NamedIndividual))

        # Data properties
        g.add((book_uri, LIB["hasTitle"], Literal(book["title"])))
        g.add((book_uri, RDFS.label, Literal(book["title"])))

        if "year" in book:
            g.add((book_uri, LIB["hasPublicationYear"], Literal(book["year"], datatype=XSD.integer)))

        if "pages" in book:
            g.add((book_uri, LIB["hasPageCount"], Literal(book["pages"], datatype=XSD.integer)))

        if "description" in book:
            g.add((book_uri, LIB["hasDescription"], Literal(book["description"])))

        # Generate ISBN (fake but unique)
        isbn = f"978-0-{book['year']}-{hash(book['id']) % 10000:04d}-{len(book['title']) % 10}"
        g.add((book_uri, LIB["hasISBN"], Literal(isbn)))

        # Object properties
        # Author
        if "author" in book:
            author_uri = LIB[book["author"]]
            g.add((book_uri, LIB["hasAuthor"], author_uri))

        # Genre - link to genre instance
        if "genre" in book:
            genre = book["genre"]
            # Use the genre instance (e.g., FantasyInstance) instead of class
            genre_instance = LIB[f"{genre}Instance"]
            g.add((book_uri, LIB["hasGenre"], genre_instance))

        # Age Group
        if "age_group" in book:
            age_group_uri = LIB[book["age_group"]]
            g.add((book_uri, LIB["forAgeGroup"], age_group_uri))

        # Format
        if "format" in book:
            format_uri = LIB[book["format"]]
            g.add((book_uri, LIB["hasFormat"], format_uri))

        # Status
        if "status" in book:
            status_uri = LIB[book["status"]]
            g.add((book_uri, LIB["hasStatus"], status_uri))

    return g


def main():
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    ontology_dir = os.path.join(project_dir, "ontology")

    base_owl_path = os.path.join(ontology_dir, "library.owl")
    populated_owl_path = os.path.join(ontology_dir, "library_populated.owl")
    populated_ttl_path = os.path.join(ontology_dir, "library_populated.ttl")

    # Load base ontology
    print(f"Loading base ontology from {base_owl_path}...")
    g = Graph()
    g.parse(base_owl_path, format="xml")
    print(f"Base ontology has {len(g)} triples")

    # Populate with books and authors
    g = populate_ontology(g)
    print(f"Populated ontology has {len(g)} triples")

    # Save populated ontology
    print(f"Saving populated ontology to {populated_owl_path}...")
    g.serialize(destination=populated_owl_path, format="xml")

    print(f"Saving Turtle format to {populated_ttl_path}...")
    g.serialize(destination=populated_ttl_path, format="turtle")

    print("Done!")

    # Print some statistics
    books_count = len(list(g.subjects(LIB["hasTitle"], None)))
    authors_count = len(list(g.subjects(LIB["hasAuthorName"], None)))
    print(f"\nStatistics:")
    print(f"  Books in ontology: {books_count}")
    print(f"  Authors in ontology: {authors_count}")


if __name__ == "__main__":
    main()
