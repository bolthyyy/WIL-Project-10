import argparse
import json

from .rag import ask
from .settings import MANUAL_METADATA_FILE


def load_vehicles() -> dict:
    with open(
        MANUAL_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        manuals = json.load(file)

    return {
        manual["vehicle_id"]: manual
        for manual in manuals
    }


def print_vehicle_list(vehicles: dict):
    print("\nAvailable vehicles:\n")

    for vehicle_id, vehicle in vehicles.items():
        print(
            f"{vehicle_id:<20} "
            f"{vehicle['make']} "
            f"{vehicle['model']} "
            f"({vehicle['year']})"
        )


def main():
    vehicles = load_vehicles()

    parser = argparse.ArgumentParser(
        description="Rental Fleet RAG Assistant"
    )

    parser.add_argument(
        "--vehicle",
        help=(
            "Vehicle ID. Leave blank to search "
            "all manuals."
        ),
    )

    parser.add_argument(
        "--question",
        help="Question to ask the RAG system.",
    )

    parser.add_argument(
        "--list-vehicles",
        action="store_true",
        help="Display all available vehicle IDs.",
    )

    arguments = parser.parse_args()

    if arguments.list_vehicles:
        print_vehicle_list(vehicles)
        return

    vehicle_id = arguments.vehicle

    if vehicle_id and vehicle_id not in vehicles:
        print(
            f"Unknown vehicle ID: {vehicle_id}"
        )
        print_vehicle_list(vehicles)
        return

    question = arguments.question

    if not question:
        question = input(
            "\nEnter customer/support question: "
        ).strip()

    if not question:
        print("No question supplied.")
        return

    vehicle_name = None

    if vehicle_id:
        vehicle = vehicles[vehicle_id]

        vehicle_name = (
            f"{vehicle['year']} "
            f"{vehicle['make']} "
            f"{vehicle['model']}"
        )

    print("\nSearching manuals...\n")

    result = ask(
        question=question,
        vehicle_id=vehicle_id,
        vehicle_name=vehicle_name,
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for source in result["sources"]:
        print(
            f"\n[Source {source['source_number']}] "
            f"{source['make']} "
            f"{source['model']} "
            f"({source['year']})"
        )

        print(
            f"Manual: {source['manual_title']}"
        )

        print(
            f"PDF page: {source['pdf_page']}"
        )

        print(
            f"Vector distance: "
            f"{source['distance']:.4f}"
        )

        print("\nRetrieved text:")
        print(source["text"])


if __name__ == "__main__":
    main()