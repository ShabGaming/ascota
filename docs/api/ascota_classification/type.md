# ascota_classification.decoration

The classification `type` is a multi-stage classification pipeline for classifying pottery types.
It uses a pre-trained DINOv2 ViT-L/14 model for feature extraction combined with optimized SVM classifiers for each stage.
The pipeline is as follows:
1. Stage 1: body vs everything else
2. Stage 2: base vs rim vs appendage
3. Stage 3: appendage subtypes using Azure OpenAI GPT-4o
---

::: ascota_classification.type
    options:
      members_order: source
      filters: ["!^__.*__$"]