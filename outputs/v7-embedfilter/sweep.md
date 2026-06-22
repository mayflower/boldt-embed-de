# v7 EmbedFilter sweep

_Candidate embedder: `outputs/v6-1-dense-top50/checkpoints/boldt-dense-rag-v6-1`. EmbedFilter basis = SVD bulk slice of the base unembedding, applied as a postprocessor. Numbers are real only if produced by a saved run._

Skipped eval sets: local_rag (no eval set defined on disk)

| eval set | role | method | dim | tau | nDCG@10 | Recall@100 | Δndcg/full | Δndcg/prefix | bytes/vec |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| webfaq_heldout | active | full | 1024 |  | 0.7046 | 0.9765 | 0.0 |  | 4096 |
| webfaq_heldout | active | prefix | 512 |  | 0.7039 | 0.9765 | -0.0007 |  | 2048 |
| webfaq_heldout | active | prefix | 256 |  | 0.7044 | 0.9714 | -0.0002 |  | 1024 |
| webfaq_heldout | active | prefix | 128 |  | 0.6958 | 0.9588 | -0.0088 |  | 512 |
| webfaq_heldout | active | prefix | 64 |  | 0.6772 | 0.9499 | -0.0274 |  | 256 |
| webfaq_heldout | active | embedfilter | 1024 | 1 | 0.7046 | 0.9765 | 0.0 |  | 4096 |
| webfaq_heldout | active | embedfilter | 512 | 2 | 0.705 | 0.981 | 0.0004 | 0.0011 | 2048 |
| webfaq_heldout | active | embedfilter | 256 | 4 | 0.6892 | 0.9702 | -0.0154 | -0.0152 | 1024 |
| webfaq_heldout | active | embedfilter | 128 | 8 | 0.672 | 0.9518 | -0.0326 | -0.0238 | 512 |
| webfaq_heldout | active | embedfilter | 64 | 16 | 0.6204 | 0.9016 | -0.0842 | -0.0568 | 256 |
| germanquad | active | full | 1024 |  | 0.8778 | 0.996 | 0.0 |  | 4096 |
| germanquad | active | prefix | 512 |  | 0.8732 | 0.996 | -0.0046 |  | 2048 |
| germanquad | active | prefix | 256 |  | 0.8549 | 0.994 | -0.0229 |  | 1024 |
| germanquad | active | prefix | 128 |  | 0.8199 | 0.988 | -0.0579 |  | 512 |
| germanquad | active | prefix | 64 |  | 0.7479 | 0.9733 | -0.1299 |  | 256 |
| germanquad | active | embedfilter | 1024 | 1 | 0.8778 | 0.996 | 0.0 |  | 4096 |
| germanquad | active | embedfilter | 512 | 2 | 0.8792 | 0.994 | 0.0014 | 0.006 | 2048 |
| germanquad | active | embedfilter | 256 | 4 | 0.8578 | 0.9933 | -0.02 | 0.0029 | 1024 |
| germanquad | active | embedfilter | 128 | 8 | 0.8148 | 0.9873 | -0.063 | -0.0051 | 512 |
| germanquad | active | embedfilter | 64 | 16 | 0.7377 | 0.9747 | -0.1401 | -0.0102 | 256 |
| dt_test | active | full | 1024 |  | 0.9748 | 0.999 | 0.0 |  | 4096 |
| dt_test | active | prefix | 512 |  | 0.9718 | 0.998 | -0.003 |  | 2048 |
| dt_test | active | prefix | 256 |  | 0.9667 | 0.998 | -0.0081 |  | 1024 |
| dt_test | active | prefix | 128 |  | 0.955 | 0.995 | -0.0198 |  | 512 |
| dt_test | active | prefix | 64 |  | 0.9203 | 0.993 | -0.0545 |  | 256 |
| dt_test | active | embedfilter | 1024 | 1 | 0.9748 | 0.999 | 0.0 |  | 4096 |
| dt_test | active | embedfilter | 512 | 2 | 0.9717 | 0.998 | -0.0031 | -0.0001 | 2048 |
| dt_test | active | embedfilter | 256 | 4 | 0.966 | 0.996 | -0.0088 | -0.0007 | 1024 |
| dt_test | active | embedfilter | 128 | 8 | 0.955 | 0.994 | -0.0198 | 0.0 | 512 |
| dt_test | active | embedfilter | 64 | 16 | 0.9053 | 0.982 | -0.0695 | -0.015 | 256 |
| gerdalir | diagnostic | full | 1024 |  | 0.0461 | 0.1848 | 0.0 |  | 4096 |
| gerdalir | diagnostic | prefix | 512 |  | 0.0436 | 0.1781 | -0.0025 |  | 2048 |
| gerdalir | diagnostic | prefix | 256 |  | 0.0417 | 0.1707 | -0.0044 |  | 1024 |
| gerdalir | diagnostic | prefix | 128 |  | 0.0341 | 0.1441 | -0.012 |  | 512 |
| gerdalir | diagnostic | prefix | 64 |  | 0.024 | 0.1187 | -0.0221 |  | 256 |
| gerdalir | diagnostic | embedfilter | 1024 | 1 | 0.0461 | 0.1848 | 0.0 |  | 4096 |
| gerdalir | diagnostic | embedfilter | 512 | 2 | 0.0469 | 0.2026 | 0.0008 | 0.0033 | 2048 |
| gerdalir | diagnostic | embedfilter | 256 | 4 | 0.0448 | 0.1963 | -0.0013 | 0.0031 | 1024 |
| gerdalir | diagnostic | embedfilter | 128 | 8 | 0.0377 | 0.1722 | -0.0084 | 0.0036 | 512 |
| gerdalir | diagnostic | embedfilter | 64 | 16 | 0.0263 | 0.1347 | -0.0198 | 0.0023 | 256 |
