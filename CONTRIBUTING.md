# Contributing

Please keep experimental manipulations isolated: one prompt variable should change at a time whenever possible.

Before a pull request:

```bash
cd backend
PYTHONPATH=app pytest tests -q
ruff check app tests
```

For frontend changes:

```bash
cd frontend/app
npm install
npm test -- --runInBand
```

Do not add RAG/vector-store dependencies unless the research question specifically requires retrieval; retrieval changes the evidence available to the tested model and would confound the current benchmark design.
