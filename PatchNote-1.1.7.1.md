# evidential output request

evidential_classification_multilabel_nologits now outputs the softmax_score and the belief_mass in the prediction file as columns

# LRODD large fix

The new interpretation of LRODD was implemented this includes big changes to datasets (introducing padded dataset) and in LRODD usage.

Perturbations are added in a similar manner as in pseudo LRODD, but it is crucial to adhere to the following specifications for fg and bg models:

```yaml

    "model": {
            "type": "generative_lstm",
            "in_features": \$protein_vocabulary_size,
            "hidden_features": \$hidden_size,
            "out_features": \$protein_vocabulary_size
        },
    "loader": {
        "type": "flexible_csv",
        "input_path": input_path,
        "input_column_names": ["target_sequence"],
        "target_column_names": ["autoencoded"]
    },
    
    // few different splitters are possible but we should
    // skip validation for optimization issue with
    // auto-regressive models
    "splitter": {
        "type": "skipping_allowed_precomputed",
        "split_path": split_path
        "splits": ...
        "skipped_columns": ["validation"]
    },
    "featurizers": [
        {
            "type": "to_index",
            "inputs": ["target_sequence"],
            "outputs": ["protein_index"],
            "should_cache": true or false,
            "vocabulary":  ["A", "C", "D", "E", .., "X"],
            "rewrite": false
        }
    ],
    "transformers": [
        {
            "type": "auto_encoder",
            "target": "autoencoded",
            "input": "protein_index" 
        }
    ],
    "inference_mode": "generative_autoencoder",
    "criterion": {
        "type": "torch.nn.CrossEntropyLoss"
    },
    "collater": {
        "type": "padded",
        "padded_column": "protein_index"
    },

    // only for background model
    "augmentations": [
        {
          "type": "protein_perturbation_sequence",
          "vocabulary":  ["A", "C", "D", "E", .., "X"],
          "p": 0.3,
          "input": "target_sequence",
          "output": "protein_index"
        }
      ],

```

Additional options are required to train auto-regressive models within kmol. For clarification, please refer to the provided configuration examples.

To perform inference with real log likelihood, use the following configuration:


```yaml
    "model": {
            "type": "lrodd",
            "model_configs": [
                {
                    classifier_model
                },                
                {
                    fg_model...
                },
                {
                    bg_model...
                }
            ]
        },
        "checkpoint_path": [
            "classifier_model",
            "fg_model_path",
            "bg_model_path"
        ],
        "inference_mode": "loss_aware",
        "collater": {
            "type": "padded",
            "padded_column": "protein_index"
        }    
```

# LRODD EDL refactor using observers

Observers are now used for easier inclusion of uncertainty modes in kmol. The refactor appears at many level and removes the need of defining in the config file "autoencoder" and "inference_mode" (modes outside mc_dropout). 
Names were also changed after team discussion.

