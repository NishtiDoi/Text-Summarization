# Text Summarization with Fine-tuned BART

A dialogue summarization system that fine-tunes Facebook's BART-large-xsum model on the SAMSum dataset to generate concise summaries of conversational text.

## Overview

This project implements an end-to-end text summarization pipeline that:
- Fine-tunes a pre-trained BART model on conversational data
- Evaluates performance using ROUGE metrics
- Provides an easy-to-use inference pipeline for generating summaries

## Dataset

The project uses the **SAMSum dataset**, which contains:
- Messenger-like conversations between friends
- Human-written summaries of these conversations
- Approximately 16,000 conversation-summary pairs

## Model Architecture

- **Base Model**: `facebook/bart-large-xsum`
- **Task**: Sequence-to-sequence text summarization
- **Framework**: Hugging Face Transformers

## Requirements

```bash
pip install transformers[sentencepiece]
pip install datasets
pip install sacrebleu
pip install rouge_score
pip install py7zr
pip install accelerate
pip install evaluate
pip install nltk
pip install matplotlib
pip install pandas
pip install torch
pip install tqdm
```

## Installation & Setup

1. **Clone and install dependencies**:
```bash
# Install required packages (see Requirements section)
```

2. **Download the dataset**:
```bash
wget https://github.com/entbappy/Branching-tutorial/raw/master/summarizer-data.zip
unzip summarizer-data.zip
```

3. **Check GPU availability**:
```bash
nvidia-smi  # Verify CUDA is available
```

## Usage

### Training

The training pipeline includes:

1. **Data Preprocessing**:
   - Loads SAMSum dataset from disk
   - Cleans dialogue text (removes artifacts like `<file_gif>`)
   - Tokenizes input dialogues and target summaries
   - Adds instruction prompt: "Summarize the following conversation:"

2. **Model Training**:
   - 3 epochs of fine-tuning
   - Batch size: 2 per device
   - Learning rate: 2e-5
   - Gradient accumulation: 8 steps
   - Evaluation every 500 steps

3. **Training Arguments**:
```python
TrainingArguments(
    output_dir='pegasus-samsum',
    num_train_epochs=3,
    warmup_steps=200,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy='steps',
    eval_steps=500
)
```

### Evaluation

The model is evaluated using ROUGE metrics:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L

### Inference

Generate summaries for new conversations:

```python
from transformers import pipeline, AutoTokenizer

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer=tokenizer)

# Generation parameters
gen_kwargs = {
    "length_penalty": 0.8, 
    "num_beams": 8, 
    "max_length": 128
}

# Generate summary
conversation = "Your dialogue text here..."
summary = pipe(conversation, **gen_kwargs)[0]["summary_text"]
print(summary)
```

## Key Features

- **Batch Processing**: Efficient evaluation with configurable batch sizes
- **GPU Support**: Automatic CUDA detection and utilization
- **Robust Tokenization**: Handles variable-length inputs with truncation
- **Comprehensive Metrics**: Multi-dimensional ROUGE evaluation
- **Model Persistence**: Save and load trained models and tokenizers

## File Structure

```
├── text_summarization.py          # Main training script
├── pegasus-samsum-model/          # Saved fine-tuned model
├── tokenizer/                     # Saved tokenizer
├── samsum_dataset/                # SAMSum dataset files
└── README.md                      # This file
```

## Training Details

### Data Processing
- Input format: "Summarize the following conversation: {dialogue}"
- Maximum input length: 1024 tokens
- Maximum summary length: 128 tokens
- Special tokens handled automatically

### Model Configuration
- **Length Penalty**: 0.8 (slightly favors shorter summaries)
- **Beam Search**: 8 beams for better quality
- **Gradient Accumulation**: 8 steps (effective batch size: 16)

## Performance

The model is evaluated on the test set using ROUGE metrics. Results are displayed in a pandas DataFrame showing:
- ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum F1-scores
- Comparison with baseline metrics

## Example Output

**Input Dialogue**:
```
Amanda: I baked cookies. Do you want some?
Jerry: Sure! What kind of cookies?
Amanda: Chocolate chip and oatmeal. 
Jerry: I'll take some chocolate chip. Thanks!
```

**Generated Summary**:
```
Amanda baked cookies and offered some to Jerry. Jerry requested chocolate chip cookies.
```

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (training tested on GPU)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: ~5GB for model files and dataset

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Ensure GPU is being utilized (`device = "cuda"`)
3. **Import Errors**: Verify all dependencies are installed with correct versions

## Contributing

Feel free to submit issues and enhancement requests. Areas for improvement:
- Experiment with different base models
- Add support for other summarization datasets
- Implement additional evaluation metrics
- Add inference optimizations

## License

This project uses models and datasets with their respective licenses:
- BART model: Apache 2.0
- SAMSum dataset: CC BY-NC-SA 4.0

## Acknowledgments

- Hugging Face Transformers library
- SAMSum dataset creators
- Facebook AI Research (BART model)
