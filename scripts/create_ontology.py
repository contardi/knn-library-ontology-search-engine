#!/usr/bin/env python3
"""
Create the base Library Ontology structure.
This script generates the OWL file with all classes, properties, and base individuals.
"""

from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Define namespaces
LIB = Namespace("http://example.org/library#")
BASE = "http://example.org/library"

def create_ontology():
    g = Graph()

    # Bind namespaces
    g.bind("lib", LIB)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Define ontology
    g.add((URIRef(BASE), RDF.type, OWL.Ontology))
    g.add((URIRef(BASE), RDFS.label, Literal("Community Library Ontology")))
    g.add((URIRef(BASE), RDFS.comment, Literal("An ontology for a community library search engine")))

    # ========== CLASSES ==========

    # Book hierarchy
    classes = {
        "Book": None,
        "FictionBook": "Book",
        "NonFictionBook": "Book",
        "Novel": "FictionBook",
        "ShortStoryCollection": "FictionBook",
        "Poetry": "FictionBook",
        "Biography": "NonFictionBook",
        "Educational": "NonFictionBook",
        "Reference": "NonFictionBook",
        "SelfHelp": "NonFictionBook",

        # Genre hierarchy
        "Genre": None,
        "FictionGenre": "Genre",
        "NonFictionGenre": "Genre",
        "Mystery": "FictionGenre",
        "Romance": "FictionGenre",
        "ScienceFiction": "FictionGenre",
        "Fantasy": "FictionGenre",
        "Thriller": "FictionGenre",
        "HistoricalFiction": "FictionGenre",
        "Horror": "FictionGenre",
        "Adventure": "FictionGenre",
        "Science": "NonFictionGenre",
        "History": "NonFictionGenre",
        "BiographyGenre": "NonFictionGenre",
        "Technology": "NonFictionGenre",
        "Philosophy": "NonFictionGenre",
        "Arts": "NonFictionGenre",
        "Nature": "NonFictionGenre",
        "Sports": "NonFictionGenre",

        # Person hierarchy
        "Person": None,
        "Author": "Person",

        # Other classes
        "AgeGroup": None,
        "Format": None,
        "AvailabilityStatus": None,
    }

    for cls, parent in classes.items():
        g.add((LIB[cls], RDF.type, OWL.Class))
        if parent:
            g.add((LIB[cls], RDFS.subClassOf, LIB[parent]))

    # Make Genre subclasses disjoint
    fiction_genres = ["Mystery", "Romance", "ScienceFiction", "Fantasy", "Thriller",
                      "HistoricalFiction", "Horror", "Adventure"]
    nonfiction_genres = ["Science", "History", "BiographyGenre", "Technology",
                         "Philosophy", "Arts", "Nature", "Sports"]

    # ========== OBJECT PROPERTIES ==========

    object_properties = [
        ("hasAuthor", "Book", "Author", None),
        ("writtenBy", "Book", "Author", None),
        ("authorOf", "Author", "Book", None),
        ("hasGenre", "Book", "Genre", None),
        ("forAgeGroup", "Book", "AgeGroup", None),
        ("hasFormat", "Book", "Format", None),
        ("hasStatus", "Book", "AvailabilityStatus", "Functional"),
        ("relatedTo", "Book", "Book", "Symmetric"),
    ]

    for prop, domain, range_, characteristic in object_properties:
        g.add((LIB[prop], RDF.type, OWL.ObjectProperty))
        g.add((LIB[prop], RDFS.domain, LIB[domain]))
        g.add((LIB[prop], RDFS.range, LIB[range_]))
        if characteristic == "Functional":
            g.add((LIB[prop], RDF.type, OWL.FunctionalProperty))
        elif characteristic == "Symmetric":
            g.add((LIB[prop], RDF.type, OWL.SymmetricProperty))

    # Inverse properties
    g.add((LIB["writtenBy"], OWL.inverseOf, LIB["authorOf"]))

    # ========== DATA PROPERTIES ==========

    data_properties = [
        ("hasTitle", "Book", XSD.string),
        ("hasISBN", "Book", XSD.string),
        ("hasPublicationYear", "Book", XSD.integer),
        ("hasPageCount", "Book", XSD.integer),
        ("hasDescription", "Book", XSD.string),
        ("hasAuthorName", "Author", XSD.string),
        ("hasBirthYear", "Author", XSD.integer),
        ("hasNationality", "Author", XSD.string),
        ("hasMinAge", "AgeGroup", XSD.integer),
        ("hasMaxAge", "AgeGroup", XSD.integer),
        ("hasAgeGroupName", "AgeGroup", XSD.string),
    ]

    for prop, domain, range_ in data_properties:
        g.add((LIB[prop], RDF.type, OWL.DatatypeProperty))
        g.add((LIB[prop], RDFS.domain, LIB[domain]))
        g.add((LIB[prop], RDFS.range, range_))

    # ========== INDIVIDUALS (AgeGroups) ==========

    age_groups = [
        ("EarlyChildhood", 0, 5, "Early Childhood (0-5)"),
        ("Children", 6, 8, "Children (6-8)"),
        ("MiddleGrade", 9, 12, "Middle Grade (9-12)"),
        ("YoungAdult", 13, 17, "Young Adult (13-17)"),
        ("Adult", 18, 99, "Adult (18+)"),
    ]

    for name, min_age, max_age, label in age_groups:
        g.add((LIB[name], RDF.type, LIB["AgeGroup"]))
        g.add((LIB[name], RDF.type, OWL.NamedIndividual))
        g.add((LIB[name], LIB["hasMinAge"], Literal(min_age, datatype=XSD.integer)))
        g.add((LIB[name], LIB["hasMaxAge"], Literal(max_age, datatype=XSD.integer)))
        g.add((LIB[name], LIB["hasAgeGroupName"], Literal(label)))
        g.add((LIB[name], RDFS.label, Literal(label)))

    # ========== INDIVIDUALS (Formats) ==========

    formats = ["Hardcover", "Paperback", "EBook", "AudioBook"]
    for fmt in formats:
        g.add((LIB[fmt], RDF.type, LIB["Format"]))
        g.add((LIB[fmt], RDF.type, OWL.NamedIndividual))
        g.add((LIB[fmt], RDFS.label, Literal(fmt)))

    # ========== INDIVIDUALS (AvailabilityStatus) ==========

    statuses = ["Available", "CheckedOut", "Reserved"]
    for status in statuses:
        g.add((LIB[status], RDF.type, LIB["AvailabilityStatus"]))
        g.add((LIB[status], RDF.type, OWL.NamedIndividual))
        g.add((LIB[status], RDFS.label, Literal(status)))

    # ========== INDIVIDUALS (Genre instances) ==========

    # Create genre instances for easier querying
    all_genres = fiction_genres + nonfiction_genres
    for genre in all_genres:
        instance_name = f"{genre}Genre" if genre not in ["BiographyGenre"] else genre
        if genre != "BiographyGenre":
            g.add((LIB[f"{genre}Instance"], RDF.type, LIB[genre]))
            g.add((LIB[f"{genre}Instance"], RDF.type, OWL.NamedIndividual))
            g.add((LIB[f"{genre}Instance"], RDFS.label, Literal(genre)))

    return g


def main():
    import os

    g = create_ontology()

    # Use paths relative to the script location (works in both host and container)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    ontology_dir = os.path.join(project_dir, "ontology")

    # Ensure ontology directory exists
    os.makedirs(ontology_dir, exist_ok=True)

    # Save to file
    output_path = os.path.join(ontology_dir, "library.owl")
    g.serialize(destination=output_path, format="xml")
    print(f"Ontology saved to {output_path}")

    # Also save in Turtle format for readability
    turtle_path = os.path.join(ontology_dir, "library.ttl")
    g.serialize(destination=turtle_path, format="turtle")
    print(f"Turtle format saved to {turtle_path}")


if __name__ == "__main__":
    main()
