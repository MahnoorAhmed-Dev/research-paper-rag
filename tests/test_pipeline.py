import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from src.ingest import rebuild_index, ingest_pdfs
from src.chain import get_answer

# 1. Confirm CLI-equivalent path still works
rebuild_index()

# 2. Confirm incremental add works (pick a real filename from data/pdfs/)
added = ingest_pdfs([Path("data/pdfs/gan.pdf")])
print(f"Added {added} chunks")

# 3. Confirm retrieval still answers correctly after both
result = get_answer("What is the main contribution of this paper?")
print(result["answer"])
print(result["sources"])