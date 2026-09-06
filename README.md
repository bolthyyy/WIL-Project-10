# Rental Fleet Roadside Support RAG Assistant

**Group ID:** WIL Project 10

## Group Members

| Student ID | Full Name |
|------------|-----------|
| s4104354 | Jayden Bolth |
| s4096381 | Jack Gigl |
| s4081433 | Ka Wang Cheng |
| s4091560 | Chad Johnson |

## Setup Instructions

These instructions are for running the baseline RAG system locally on Windows.

### 1. Clone the repository

```powershell
git clone https://github.com/bolthyyy/WIL-Project-10.git
cd WIL-Project-10
```

### 2. Install Python

Python 3.12 or 3.13 is recommended.

Check whether Python is installed:

```powershell
py --list
```

### 3. Create a virtual environment

From the project root:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your terminal should now begin with:

```text
(.venv)
```

### 4. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Install Ollama

Download and install Ollama from:

https://ollama.com/

Then install the model used by the baseline system:

```powershell
ollama pull gemma3:4b
```

Check that it works:

```powershell
ollama run gemma3:4b
```

Exit the interactive Ollama session once it responds.

### 6. Check the vehicle manuals

The following PDFs should be located in:

```text
data/manuals/
```

Required files:

```text
Toyota_Corolla.pdf
Toyota_RAV4.pdf
Mazda_3.pdf
Mazda_CX5.pdf
Hyundai_i30.pdf
Hyundai_Tucson.pdf
Kia_Cerato.pdf
Kia_Sportage.pdf
```

The metadata for these manuals is stored in:

```text
data/manuals.json
```

### 7. Check PDF text extraction

Run:

```powershell
python -m src.check_pdfs
```

Each manual should report a page count and extracted characters.

If a manual reports no extracted text, do not continue with that PDF until the issue is resolved.

### 8. Build the vector database

Run:

```powershell
python -m src.ingest --reset
```

This will:

- extract text from all manuals
- split the manuals into chunks
- create embeddings
- store the chunks and vehicle metadata in Chroma

The generated database will be stored in:

```text
chroma_db/
```

The database is generated locally and is not committed to GitHub.

### 9. List available vehicle IDs

```powershell
python -m src.cli --list-vehicles
```

Example vehicle IDs include:

```text
toyota_corolla
toyota_rav4
mazda_3
mazda_cx5
hyundai_i30
hyundai_tucson
kia_cerato
kia_sportage
```

### 10. Ask the RAG system a question

Example:

```powershell
python -m src.cli --vehicle toyota_corolla --question "How do I open the fuel filler door?"
```

The system should display:

- a generated answer
- cited source numbers
- the retrieved manual passages
- vehicle information
- PDF page numbers
- vector distances

### 11. Search across all manuals

To test retrieval without specifying a vehicle:

```powershell
python -m src.cli --question "How do I reset the tyre pressure monitoring system?"
```

This searches all vehicle manuals rather than filtering to one vehicle.

## Running the Project Later

After the initial setup, you normally only need to:

```powershell
cd WIL-Project-10
.\.venv\Scripts\Activate.ps1
```

Then run a query:

```powershell
python -m src.cli --vehicle toyota_corolla --question "Your question here"
```

You do **not** need to rebuild the Chroma database every time.

Re-run:

```powershell
python -m src.ingest --reset
```

only if the manuals, embedding model, chunking settings, or ingestion code have changed.

## Current Baseline

The baseline currently uses:

- `pypdf` for PDF text extraction
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Chroma as the local vector database
- `gemma3:4b` through Ollama for answer generation
- vehicle metadata filtering
- 220-word chunks
- 40-word chunk overlap
- Top 5 retrieval

Please keep the baseline configuration unchanged when conducting experiments unless the change itself is being evaluated.
```