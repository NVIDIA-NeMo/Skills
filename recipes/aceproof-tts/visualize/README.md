# AceProof TTS Visualization

Lightweight web UI for browsing a single AceProof-TTS run.

## Run

```bash
pip install fastapi uvicorn
python /lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/recipes/aceproof-tts/visualize/server.py \
  --run-dir /lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/aceproof-tts-debug4 \
  --host 0.0.0.0 --port 8000
```

Then open in your browser:
```
http://localhost:8000
```

If running on a login node, use port forwarding:

```bash
ssh -L 8000:localhost:8000 <login-node>
```

## Notes

- Proof trends are computed on the fly from `proof_pool`.
- Verifications are loaded per round from `rounds/R*/verify/output.jsonl`.
