from __future__ import unicode_literals, print_function, division
import re
import json
import time
import math
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from model import CurriculumSeqEncoder, CurriculumSeqDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2

hidden_size = 512
MAX_LENGTH = 100
MIN_FREQ = 1
data_dir = 'raw_data.json'
batch_size = 64
learning_rate = 0.001
num_epochs = 50
teacher_forcing_ratio = 0.5  

class Lang:
    def __init__(self):
        self.word2index = {"<SOS>": SOS_token, "<EOS>": EOS_token, "<PAD>": PAD_token}
        self.word2count = {}
        self.index2word = {SOS_token: "<SOS>", EOS_token: "<EOS>", PAD_token: "<PAD>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class TypeLang:
    def __init__(self):
        self.type2index = {}
        self.index2type = {}
        self.n_types = 0

    def add_type(self, type_str):
        type_str = type_str.lower()
        if type_str not in self.type2index:
            self.type2index[type_str] = self.n_types
            self.index2type[self.n_types] = type_str
            self.n_types += 1

    def get_type_index(self, type_str):
        return self.type2index.get(type_str.lower(), None)

    def get_type_str(self, index):
        return self.index2type.get(index, "<UNK>")

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    return text

def tokenize(sentences):
    return [sentence.split(' ') for sentence in sentences]

def build_vocab(tokenized_sentences, lang_obj, min_freq=1):
    for sentence in tokenized_sentences:
        lang_obj.add_sentence(' '.join(sentence))
    words_to_remove = [word for word, count in lang_obj.word2count.items() if count < min_freq]
    for word in words_to_remove:
        if word not in ["<SOS>", "<EOS>", "<PAD>"]:
            del lang_obj.word2index[word]
            del lang_obj.word2count[word]
            index = [k for k, v in lang_obj.index2word.items() if v == word][0]
            del lang_obj.index2word[index]
            lang_obj.n_words -= 1
    return lang_obj.word2index

def encode_sentences(tokenized_sentences, vocab):
    encoded = []
    for sentence in tokenized_sentences:
        encoded_sentence = [vocab.get(word, PAD_token) for word in sentence]
        encoded_sentence.append(EOS_token)
        encoded.append(encoded_sentence)
    return encoded

def pad_sequences(sequences, max_length, pad_value=PAD_token):
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            seq = seq + [pad_value] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
            if seq[-1] != EOS_token:
                seq[-1] = EOS_token
        padded.append(seq)
    return padded

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f'{as_minutes(s)} (- {as_minutes(rs)})'

def train_model(encoder, decoder, dataloader, encoder_optimizer, decoder_optimizer, criterion_gen, criterion_cls, epochs=10):
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        start_time = time.time()
        total_loss = 0
        total_cls_loss = 0
        total_gen_loss = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}', unit='batch')
        for batch_idx, (input_tensor, target_tensor, input_lengths, type_tensor) in enumerate(progress_bar):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            input_lengths = input_lengths.to(device)
            type_tensor = type_tensor.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden, class_logits = encoder(input_tensor, input_lengths)

            cls_loss = criterion_cls(class_logits, type_tensor)

            decoder_outputs, decoder_hidden, _ = decoder(
                encoder_outputs, encoder_hidden, target_tensor, teacher_forcing_ratio=teacher_forcing_ratio, type_indices=type_tensor
            )

            loss_gen = criterion_gen(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            loss = loss_gen + cls_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_gen_loss += loss_gen.item()

            progress_bar.set_postfix({'Total Loss': loss.item(), 'CLS Loss': cls_loss.item(), 'GEN Loss': loss_gen.item()})

        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_gen_loss = total_gen_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f'Epoch {epoch} completed in {as_minutes(elapsed)} with average loss {avg_loss:.4f} (CLS: {avg_cls_loss:.4f}, GEN: {avg_gen_loss:.4f})')

        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': avg_loss,
        }, f'./checkpoints/model_epoch_{epoch}.pth')

def evaluate(encoder, decoder, sentence, input_lang, type_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        sentence = normalize_text(sentence)
        tokenized_sentence = sentence.split(' ')
        encoded_sentence = [input_lang.get(word, PAD_token) for word in tokenized_sentence]
        encoded_sentence.append(EOS_token)
        input_padded = pad_sequences([encoded_sentence], max_length)
        input_tensor = torch.tensor(input_padded, dtype=torch.long).to(device)

        input_length = min(len(encoded_sentence), max_length)
        input_lengths = torch.tensor([input_length], dtype=torch.long).to(device)

        encoder_outputs, encoder_hidden, class_logits = encoder(input_tensor, input_lengths)

        _, predicted_type = torch.max(class_logits, 1)
        predicted_type = predicted_type.item()

        decoder_input = torch.full((1, 1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        attentions = []

        type_indices = torch.tensor([predicted_type], dtype=torch.long).to(device)

        for i in range(max_length):
            embedded = decoder.embedding(decoder_input)
            query = decoder_hidden[0][-1]
            context, attn_weights = decoder.attention(query, encoder_outputs)
            context = context.unsqueeze(1)
            type_step = decoder.type_embedding(type_indices).unsqueeze(1)
            gru_input = torch.cat((embedded, context, type_step), dim=2)
            output, decoder_hidden = decoder.gru(gru_input, decoder_hidden)
            output = decoder.out(output.squeeze(1))
            output = F.log_softmax(output, dim=1)
            topv, topi = output.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_word = output_lang.get(topi.item(), "<UNK>")
                decoded_words.append(decoded_word)
            decoder_input = topi.detach()

        return ' '.join(decoded_words), type_lang.get_type_str(predicted_type)

def main():
    data = load_data(data_dir)
    questions = [normalize_text(item['question']) for item in data]
    responses = [normalize_text(item['response']) for item in data]
    types = [item['type'] for item in data]

    tokenized_questions = tokenize(questions)
    tokenized_responses = tokenize(responses)

    input_lang = Lang()
    output_lang = Lang()
    type_lang = TypeLang()

    build_vocab(tokenized_questions, input_lang, min_freq=MIN_FREQ)
    build_vocab(tokenized_responses, output_lang, min_freq=MIN_FREQ)

    for type_str in types:
        type_lang.add_type(type_str)
    num_types = type_lang.n_types
    print(f"Number of Types: {num_types}")

    print(f"Input Language Vocabulary Size: {input_lang.n_words}")
    print(f"Output Language Vocabulary Size: {output_lang.n_words}")

    encoded_questions = encode_sentences(tokenized_questions, input_lang.word2index)
    encoded_responses = encode_sentences(tokenized_responses, output_lang.word2index)
    encoded_types = [type_lang.get_type_index(t) for t in types]

    input_padded = pad_sequences(encoded_questions, MAX_LENGTH)
    target_padded = pad_sequences(encoded_responses, MAX_LENGTH)

    input_lengths = [min(len(seq), MAX_LENGTH) for seq in encoded_questions]
    target_lengths = [min(len(seq), MAX_LENGTH) for seq in encoded_responses]
    difficulty = [input_len + target_len for input_len, target_len in zip(input_lengths, target_lengths)]

    sorted_indices = sorted(range(len(difficulty)), key=lambda k: difficulty[k])
    input_padded = [input_padded[i] for i in sorted_indices]
    target_padded = [target_padded[i] for i in sorted_indices]
    encoded_types = [encoded_types[i] for i in sorted_indices]

    input_tensor = torch.tensor(input_padded, dtype=torch.long)
    target_tensor = torch.tensor(target_padded, dtype=torch.long)
    input_lengths = torch.tensor([min(len(seq), MAX_LENGTH) for seq in encoded_questions], dtype=torch.long)
    type_tensor = torch.tensor(encoded_types, dtype=torch.long)

    train_data = TensorDataset(input_tensor, target_tensor, input_lengths, type_tensor)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    encoder = CurriculumSeqEncoder(input_lang.n_words, hidden_size, num_layers=2, dropout_p=0.3, bidirectional=True, num_types=num_types).to(device)
    decoder = CurriculumSeqDecoder(hidden_size, output_lang.n_words, num_layers=1, dropout_p=0.1, num_types=num_types).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion_gen = nn.NLLLoss(ignore_index=PAD_token)
    criterion_cls = nn.CrossEntropyLoss()

    train_model(encoder, decoder, train_dataloader, encoder_optimizer, decoder_optimizer, criterion_gen, criterion_cls, epochs=num_epochs)

    encoder.eval()
    decoder.eval()

    print("\nTraining completed! You can now interact with the chatbot.")
    while True:
        try:
            input_sentence = input("You: ")
            if input_sentence.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Goodbye! Have a great day!")
                break
            response, response_type = evaluate(encoder, decoder, input_sentence, input_lang.word2index, type_lang, output_lang.index2word)
            print(f"Chatbot ({response_type}): {response}")
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye! Have a great day!")
            break

if __name__ == "__main__":
    main()
