# Geophysical Inversion with In-Context Learning

This repository uses in-context learning (ICL) with a large language model (LLM) for one-dimensional time-domain airborne electromagnetic (AEM) inversion. Given the response curve and flight height for a single sounding, the program retrieves similar responses from an example library and supplies them to the LLM along with their corresponding resistivity models. The LLM then proposes a layered resistivity model for the sounding.

## Preparing for Inversion

### LLM Service

Configure `BASE_URL`, `MODEL`, and `api_key` for an OpenAI-compatible service in [`ICL/clients.py`](ICL/clients.py). The client appends `/v1/` to `BASE_URL` and sends requests with `response_format={"type": "json_object"}` and `reasoning_effort="none"`. The selected endpoint and model must accept these parameters and return a JSON object containing a `rho_code` field.

Update `api_key` to read from a local environment variable, for example by importing `os` and using `os.environ["AEM_LLM_API_KEY"]`. Each request sends the encoded sounding response, flight height, and retrieved examples to the configured service.

### Building the Example Library

Place `data.npy`, `h.npy`, and `pre.npy` in the repository root. You can change the input paths and output directory in [`get_data.py`](get_data.py). The script writes to `tests/filed/` by default and may overwrite existing library files with the same names.

```bash
python get_data.py
```

### Running Inversion

After preparing the data and configuring API access, run the structurally constrained inversion from the repository root.

```bash
python inversion.py
```

To run the iterative inversion, use the following command.

```bash
python -c "from inversion import run_loop; run_loop()"
```
