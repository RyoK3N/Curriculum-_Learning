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
To clone the repository

```bash
git clone https://github.com/RyoK3N/Curriculum-_Learning/blob/main/Curriculum-Seq2Seq-Att
```

after cloning,
install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

## How to Run the Project

1. Prepare the Data
Ensure that your data is in a JSON format similar to the following structure:

```json
[
    {
        "question": "Your question here",
        "response": "The corresponding response",
        "type": "The type of question"
    }
]
```
Save the file as __raw_data.json__ in the project directory.

2. Train the Model
Run the following command to start training the model:

```bash
python main.py
```

This will train the chatbot model for 50 epochs using the default settings. The model checkpoints will be saved in the __./checkpoints/__ directory.

3. Interact with the Chatbot
Once the model is trained, you can interact with the chatbot:

```bash
python main.py
```
Type your queries, and the chatbot will generate responses based on the trained model. Type __exit__, __quit__, or __bye__ to end the session.

## Hyperparameters
```vbnet
hidden_size: 512 (Size of the LSTM hidden layers)
MAX_LENGTH: 100 (Maximum sequence length)
batch_size: 64 (Batch size during training)
learning_rate: 0.001 (Learning rate for the optimizer)
num_epochs: 50 (Number of training epochs)
teacher_forcing_ratio: 0.5 (Ratio for teacher forcing during training)
dropout_p: 0.3 (Dropout probability in encoder/decoder layers)
```
## Training and Evaluation
During training, the model computes:

**Generation Loss**: Negative log likelihood loss for the generated sequences.

**Classification Loss**: Cross-entropy loss for the type classification.

Both losses are backpropagated to update the encoder and decoder parameters.

### Example Output
Here is an example interaction with the trained chatbot:

```css
You: What is the weather like today?
Chatbot (weather): It's sunny with a chance of rain later in the afternoon.
```

## Model Checkpoints
The model automatically saves checkpoints after every epoch. You can load these checkpoints for evaluation or further training.

## Customization

**Curriculum Learning:** You can modify the difficulty measure used to sort the training examples in main.py.

**Type Classification:** You can adjust the number of predefined types and their categories by updating the TypeLang class.

**Vocabulary:** Modify the MIN_FREQ parameter to filter out rare words from the vocabulary.

## Contributing
Feel free to contribute to the project by submitting issues or pull requests. Before making contributions, ensure that your code adheres to the style of the existing codebase and includes appropriate documentation.

## License

```css
This project is licensed under the MIT License.
```
