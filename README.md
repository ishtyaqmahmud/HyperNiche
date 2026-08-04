# HyperNiche
Learning heterophilic cellular niches from spatial transcriptomics using adaptive hypergraph neural networks.


graph TD
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef model fill:#fff3e0,stroke:#e65100,stroke-width:3px;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef analysis fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph Inputs [1. Spatial Transcriptomics Data]
        X[Gene Expression Matrix] ::: input
        P[Cell Centroid Coordinates] ::: input
        Y[Reference Cell-Type Labels <br> For Training] ::: input
    end

    HN{HyperNiche Framework <br> End-to-End Blackbox} ::: model

    subgraph Outputs [2. Direct Model Outputs]
        Emb[Learned Cell Embeddings] ::: output
        Niches[Learned Cellular Niches <br> Soft Hyperedges] ::: output
    end

    subgraph Evaluation [3. Biological & Structural Evaluation]
        Clust[Cell-Type Recovery <br> K-Means, ARI, NMI] ::: analysis
        Struct[Structural Validation <br> Size, Spatial Radius, Entropy] ::: analysis
        Bio[Biological Enrichment <br> Local Null vs Global Null] ::: analysis
    end

    %% Connections
    X --> HN
    P --> HN
    Y -. Supervision .-> HN
    
    HN ==> Emb
    HN ==> Niches
    
    Emb --> Clust
    Niches --> Struct
    Niches --> Bio
