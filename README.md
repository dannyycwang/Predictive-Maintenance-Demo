# Predictive-Maintenance-Demo

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Online mode entrypoint:

```bash
streamlit run streamlit_app_online.py
```

## OpenAI API key (do **NOT** put key in GitHub code)

This app reads OpenAI settings in this order:
1. `st.secrets[...]` (Streamlit/Cloud secrets)
2. environment variables
3. default values in code

Supported keys:
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_API_ENDPOINT` (optional)

### If server is on GitHub

Use **repository / platform secrets**, not plaintext in repo:

- **Streamlit Community Cloud**: App → Settings → Secrets, add:
  ```toml
  OPENAI_API_KEY = "sk-..."
  OPENAI_MODEL = "gpt-4o-mini"
  OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
  ```
- **GitHub Actions / self-hosted deploy**: add Repository Secret `OPENAI_API_KEY`, and inject as env var at deploy runtime.

Never commit `.env`, `secrets.toml`, or API keys into git history.


Note: some code-review platforms cannot handle binary patches well; this project uses SVG (text) for the Duval figure to avoid binary-file upload issues.
