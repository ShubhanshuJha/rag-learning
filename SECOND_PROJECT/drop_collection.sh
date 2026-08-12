docker compose exec api python3 -c "
from app.services import vector_store
with vector_store.get_client() as client:
    if client.collections.exists('DocChunks'):
        client.collections.delete('DocChunks')
        print('Dropped old collection')
"
docker compose restart api

