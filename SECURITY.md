# Security

Do not commit API keys or `.env` files.

The default benchmark uses a local Ollama endpoint. If you add a remote provider adapter, load credentials from environment variables and never return those credentials through the API or frontend.
