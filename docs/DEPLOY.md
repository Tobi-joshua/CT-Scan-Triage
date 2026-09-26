# Streamlit deployment

The app is structured for Streamlit Community Cloud:

- repository: `Tobi-joshua/CT-Scan-Triage`
- entrypoint: `app.py`
- Python dependencies: `requirements.txt`
- configuration: `.streamlit/config.toml`

A trained checkpoint must exist at `artifacts/cxr_mobilenetv3.pt` before the public demo can return predictions. Do not deploy an unvalidated/random checkpoint.

After a checkpoint has been trained and evaluated, add the checkpoint (or a controlled download mechanism) and deploy from Streamlit Community Cloud. The UI must retain the research-only / non-diagnostic warning.
