# Project Pipeline Mermaid Diagram

```mermaid
flowchart TD
    A[Start Project] --> B{Data Source}

    B --> C[BVBRC phenotype data<br/>genome_amr endpoint]
    B --> D[NCBI AST Browser export<br/>asts.tsv / ecoli_resistance_ncbi.csv]

    C --> E[BVBRC preprocessing<br/>clean labels, deduplicate, split by standard]
    E --> F[BVBRC gene annotations<br/>CARD + NDARO]
    F --> G[BVBRC merged modeling set]

    D --> H[NCBI preprocessing<br/>filter E. coli, keep ciprofloxacin R/S]
    H --> I[Map BioSample to Assembly<br/>biosample_to_assembly.csv]
    I --> J[Download genome FASTA files]
    J --> K[Run AMRFinderPlus]
    K --> L[Collect hits + mutation_all outputs]
    L --> M[Build quinolone-focused feature matrix<br/>gyrA, parC, parE, qnr, etc.]

    G --> N[Train ML models]
    M --> N

    N --> O[Logistic Regression]
    N --> P[Random Forest]
    N --> Q[XGBoost]
    N --> R[Optional ensembles]

    O --> S[Evaluate on held-out test data]
    P --> S
    Q --> S
    R --> S

    S --> T{Validation strategy}
    T --> U[Random isolate split]
    T --> V[BioProject holdout]
    T --> W[Leave-one-BioProject-out]

    U --> X[Metrics + ROC + confusion matrices]
    V --> X
    W --> X

    X --> Y[Results organized by source]
    Y --> Z1[results/BVBRC/...]
    Y --> Z2[results/NCBI/...]
```

## Notes

- The original project started with the BVBRC workflow.
- The current ciprofloxacin-focused analysis emphasizes the NCBI plus AMRFinder workflow.
- The stricter validation path uses BioProject-aware splitting to test generalization beyond a simple random isolate split.
