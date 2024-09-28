# Curriculum-Based Chatbot with Sequence-to-Sequence Attention Model

This project implements a curriculum-based sequence-to-sequence (Seq2Seq) chatbot using an LSTM encoder-decoder architecture with attention mechanism. The model also incorporates a type classifier, enabling the chatbot to categorize user inputs into specific predefined types and generate responses based on these categories. The project is built using PyTorch.

## Features
- **Encoder-Decoder Architecture**: Utilizes LSTM-based encoder and decoder with attention mechanism for generating sequences.
- **Attention Mechanism**: Improves performance by allowing the model to focus on specific parts of the input sequence.
- **Type Classification**: Classifies input sequences into predefined types and tailors responses accordingly.
- **Curriculum Learning**: Trains the model by sorting training examples based on sequence difficulty.
- **Teacher Forcing**: The decoder leverages teacher forcing during training to improve convergence.

## Model Architecture

1. **CurriculumSeqEncoder**: Bi-directional LSTM encoder with type classification head and layer normalization.
2. **CurriculumSeqAttention**: Implements attention to help the decoder focus on relevant encoder outputs.
3. **CurriculumSeqDecoder**: LSTM-based decoder that generates a sequence of tokens as the response while incorporating the type classification as an additional input.

## Dependencies
To install the required dependencies, you can use the following command:
```bash
pip install -r requirements.txt

