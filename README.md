# Text Summarization with PEGASUS

A text summarization project using the PEGASUS model fine-tuned on the SAMSum dataset for dialogue summarization.

## Overview

This project implements an abstractive text summarization system using Google's PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence) model. The model is fine-tuned on the SAMSum dataset, which contains conversational dialogues and their corresponding summaries.

## Features

- Fine-tuning PEGASUS model on dialogue data
- Batch processing for efficient training and evaluation
- ROUGE metric evaluation for model performance assessment
- Model and tokenizer persistence for future use
- GPU acceleration support

## Requirements

```bash
pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr
pip install --upgrade accelerate
pip install torch
pip install matplotlib pandas nltk tqdm
```

## Dataset

The project uses the SAMSum dataset, which contains:
- **Training set**: Conversational dialogues with summaries
- **Validation set**: Used for model evaluation during training
- **Test set**: Used for final performance evaluation

## Model Architecture

- **Base Model**: `google/pegasus-cnn_dailymail`
- **Model Type**: Sequence-to-Sequence (Seq2Seq) for abstractive summarization
- **Tokenizer**: PEGASUS tokenizer with support for dialogue formatting

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size (Train) | 1 |
| Batch Size (Eval) | 1 |
| Gradient Accumulation Steps | 16 |
| Max Input Length | 1024 tokens |
| Max Output Length | 128 tokens |
| Weight Decay | 0.01 |
| Warmup Steps | 500 |

## Usage

### 1. Setup and Installation

```python
# Install required packages
!pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
!pip install --upgrade accelerate

# Import necessary libraries
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
```

### 2. Load Pre-trained Model

```python
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
```

### 3. Data Preprocessing

The `convert_examples_to_features()` function handles:
- Tokenization of input dialogues (max 1024 tokens)
- Tokenization of target summaries (max 128 tokens)
- Attention mask generation

### 4. Training

```python
from transformers import Trainer, TrainingArguments

# Configure training arguments
trainer_args = TrainingArguments(
    output_dir='pegasus-samsum',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    # ... other parameters
)

# Train the model
trainer = Trainer(model=model_pegasus, args=trainer_args, ...)
trainer.train()
```

### 5. Evaluation

The project uses ROUGE metrics for evaluation:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L

### 6. Inference

```python
# Load trained model
pipe = pipeline("summarization", 
                model="pegasus-samsum-model",
                tokenizer=tokenizer)

# Generate summary
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
summary = pipe(dialogue_text, **gen_kwargs)[0]["summary_text"]
```

## File Structure

```
├── text_summarization.py          # Main training script
├── pegasus-samsum-model/           # Saved model directory
├── tokenizer/                      # Saved tokenizer directory
├── samsum_dataset/                 # Dataset directory
└── README.md                       # This file
```

## Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `length_penalty` | 0.8 | Penalty for longer sequences |
| `num_beams` | 8 | Number of beams for beam search |
| `max_length` | 128 | Maximum summary length |

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: ~5GB for model weights and dataset

## Performance

The model is evaluated using ROUGE metrics on the test set. Results are displayed in a pandas DataFrame showing:
- ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum F1 scores
- Performance comparison against baseline models

## Key Functions

### `convert_examples_to_features(example_batch)`
Converts raw dialogue and summary text into tokenized features suitable for training.

### `generate_batch_sized_chunks(list_of_elements, batch_size)`
Utility function for processing data in batches to manage memory usage.

### `calculate_metric_on_test_ds(...)`
Evaluates model performance on test dataset using ROUGE metrics with batch processing.

## Model Saving and Loading

```python
# Save model and tokenizer
model_pegasus.save_pretrained("pegasus-samsum-model")
tokenizer.save_pretrained("tokenizer")

# Load saved components
tokenizer = AutoTokenizer.from_pretrained("/content/tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("pegasus-samsum-model")
```

## Example Output

```
Dialogue:
[Sample conversation between multiple speakers]

Reference Summary:
[Human-written summary of the conversation]

Model Summary:
[AI-generated summary from the trained model]
```

## Future Improvements

- Experiment with different learning rates and batch sizes
- Try other pre-trained models (T5, BART)
- Implement early stopping based on validation loss
- Add more sophisticated evaluation metrics
- Fine-tune on domain-specific datasets

## License

This project uses models and datasets that may have their own licensing terms. Please check:
- [Hugging Face Transformers License](https://github.com/huggingface/transformers/blob/main/LICENSE)
- [SAMSum Dataset License](https://huggingface.co/datasets/samsum)
- [PEGASUS Model License](https://huggingface.co/google/pegasus-cnn_dailymail)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Slow training**: Ensure GPU is being used and consider mixed precision training
3. **Poor summaries**: Try adjusting generation parameters or training for more epochs

### Dependencies Issues

If you encounter package conflicts, try:
```bash
pip uninstall -y transformers accelerate
pip install transformers accelerate
```

## References

- [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
- [SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization](https://arxiv.org/abs/1911.12237)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
