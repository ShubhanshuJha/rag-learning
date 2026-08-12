# VALIDATION INDEXING
docker compose exec api python3 -c "
from app.services import vector_store
with vector_store.get_client() as client:
    print('Len:', len(vector_store.existing_hashes_for_doc(client, 'aws-dms-sample-v1')))
"


# ASK
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AWS DMS used for?"}'

