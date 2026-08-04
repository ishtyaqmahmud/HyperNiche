# HyperNiche
Learning heterophilic cellular niches from spatial transcriptomics using adaptive hypergraph neural networks.


graph TD
    %% Styling
    classDef input fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef matrix fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef loss fill:#ffebee,stroke:#f44336,stroke-width:2px;

    subgraph Inputs
        X[Gene Expression Matrix] ::: input
        P[Spatial Coordinates] ::: input
    end

    subgraph Phase_1_Encoding [Phase 1: Initial Encoding & Roles]
        X --> Encoder[Encoder MLP] ::: process
        Encoder --> H0[Initial Cell Embeddings] ::: matrix
        H0 --> Wm[Member Role Projection] ::: process
        H0 --> Wa[Anchor Role Projection] ::: process
    end

    subgraph Phase_2_Spatial_Compatibility [Phase 2: Spatial & Compatibility]
        P --> KNN[Spatial k-NN Graph] ::: process
        KNN --> Sij[Relative Spatial Features] ::: matrix
        Wm --> Q[Bilinear Compatibility Score] ::: process
        Wa --> Q
        Sij --> Q
        Wa --> Wj[Hyperedge Weights Prediction] ::: process
    end

    subgraph Phase_3_Hypergraph [Phase 3: Hypergraph Construction]
        Q --> Sigmoid[Sigmoid Activation] ::: process
        Sigmoid --> M[Learned Soft Incidence Matrix] ::: matrix
    end

    subgraph Phase_4_HGNN [Phase 4: Message Passing]
        H0 --> HGNN[Hypergraph Conv Layers] ::: process
        M --> HGNN
        Wj --> HGNN
        HGNN --> HL[Final Cell Representations] ::: matrix
    end

    subgraph Phase_5_Output_Loss [Phase 5: Output & Regularization]
        HL --> Classifier[Linear Classifier] ::: process
        Classifier --> Preds[Cell-Type Predictions] ::: matrix
        Preds --> Lsup[Cross-Entropy Loss] ::: loss
        
        M --> Reg[Structural Regularization] ::: process
        Wj --> Reg
        Reg --> Lreg[Sparsity, Entropy, Degeneracy & Overlap Penalty] ::: loss
    end
