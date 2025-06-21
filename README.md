# simple-rag-python
Python RAG microservice to query documents

## Run

### First time setup

```
docker compose up -d rag-api-ollama
docker compose exec rag-api-ollama ollama pull nomic-embed-text
```

### Run services

To run the API just do

```bash
docker compose up
```

The API will run on port `8000` and the documentation can be found at `http://localhost:8000/docs`.

You can also verify the ollama with 
```bash
curl http://localhost:11434/api/tags
```
