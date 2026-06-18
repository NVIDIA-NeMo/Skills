# NTT-COMPLEX

`ntt-complex` is the umbrella benchmark for compositional NemotronTranscribe
tasks. The bootstrap subtest is `ntt-complex.format_ast`, which evaluates audio
speech translation with strict output formatting.

The source data is expected to be prepared NeMo-Skills FLEURS and CoVoST2 AST/ST
manifests, matching the canary-dev burst-eval path that prepares FLEURS and
CoVoST2 as separate ASR and AST tasks. This checkout currently names those
speech-translation subsets `fleurs.st` and `covost2.st`; canary-dev references
them as `fleurs.ast` and `covost2.ast`. The prepare script accepts both layouts.

Example:

```bash
ns prepare_data fleurs covost2 --data_dir <skills_data>
NTT_COMPLEX_SOURCE_DATA_DIR=<skills_data> ns prepare_data ntt-complex --data_dir <skills_data>
ns eval --benchmarks ntt-complex.format_ast --data_dir <skills_data> ...
```

`format_ast` renders each selected source row into several strict-output prompt
variants:

- `json_object`
- `srt_single_cue`
- `markdown_table`

The evaluator extracts the translated text from the required structure, scores
format validity, and then scores the extracted translation with the standard
audio translation BLEU path.
