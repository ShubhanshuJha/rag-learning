import json
import sys
import time
import urllib.request

API_URL = "http://localhost:8000/ingest"
RETRY_DELAY_SECONDS = 10
BETWEEN_CALL_DELAY_SECONDS = 2


def ingest_once(pdf_path: str, doc_title: str) -> dict:
    boundary = "----ingestdriverboundary"
    filename = pdf_path.split("/")[-1]

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="doc_title"\r\n\r\n'
        f"{doc_title}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/ingest_large_pdf.py <pdf_path> <doc_title>")
        sys.exit(1)

    pdf_path, doc_title = sys.argv[1], sys.argv[2]
    call_number = 0
    total_inserted = 0

    while True:
        call_number += 1
        print(f"\n--- Call {call_number} ---")

        try:
            result = ingest_once(pdf_path, doc_title)
        except Exception as exc:
            print(f"Call failed: {exc}")
            print(f"Retrying in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)
            continue

        total_inserted += result["chunks_created"]
        print(
            f"doc_id={result['doc_id']} "
            f"inserted_this_call={result['chunks_created']} "
            f"skipped_duplicate={result['chunks_skipped_duplicate']} "
            f"remaining={result['chunks_remaining']} "
            f"total_inserted_so_far={total_inserted}"
        )

        if result["chunks_remaining"] == 0:
            print("\nIngestion complete — every chunk is stored.")
            break

        time.sleep(BETWEEN_CALL_DELAY_SECONDS)


if __name__ == "__main__":
    main()
