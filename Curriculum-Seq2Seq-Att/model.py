import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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

class CurriculumSeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_p=0.3, bidirectional=True, num_types=10):
        super(CurriculumSeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_types = num_types

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0
        )

        if self.bidirectional:
            self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.output_projection = nn.Linear(hidden_size * 2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_types)

    def forward(self, input_seq, input_lengths):
        embedded = self.dropout(self.embedding(input_seq))  
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.gru(packed)  
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=MAX_LENGTH)

        if self.bidirectional:
            outputs = self.output_projection(outputs)  
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  
            cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)      
            hidden = self.hidden_projection(hidden)  
            cell = self.cell_projection(cell)        
            hidden = hidden.unsqueeze(0)            
            cell = cell.unsqueeze(0)                
        else:
            hidden = hidden  
            cell = cell      

        hidden = self.layer_norm(hidden)
        cell = self.layer_norm(cell)
        cls_hidden = hidden[-1]  
        class_logits = self.classifier(cls_hidden)  

        return outputs, (hidden, cell), class_logits

class CurriculumSeqAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CurriculumSeqAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        query = query.unsqueeze(1)  
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  
        scores = scores.squeeze(2)  
        weights = F.softmax(scores, dim=1)  
        context = torch.bmm(weights.unsqueeze(1), keys)  
        context = context.squeeze(1)  
        return context, weights

class CurriculumSeqDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout_p=0.3, num_types=10):
        super(CurriculumSeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_types = num_types  

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = CurriculumSeqAttention(hidden_size)

        self.type_embedding = nn.Embedding(num_types, hidden_size)
        
        self.gru = nn.LSTM(
            hidden_size + hidden_size + hidden_size,  
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ratio=0.5, type_indices=None):

        hidden, cell = encoder_hidden  

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=encoder_outputs.device)

        decoder_outputs = []
        attentions = []

        type_embedded = self.type_embedding(type_indices)  
        type_embedded = type_embedded.unsqueeze(1).repeat(1, MAX_LENGTH, 1)  

        for i in range(MAX_LENGTH):
            embedded = self.dropout(self.embedding(decoder_input))  
            query = hidden[-1]  
            context, attn_weights = self.attention(query, encoder_outputs)  
            context = context.unsqueeze(1)  
            type_step = type_embedded[:, i, :].unsqueeze(1)  
            gru_input = torch.cat((embedded, context, type_step), dim=2)  
            output, (hidden, cell) = self.gru(gru_input, (hidden, cell))  
            output = self.out(output.squeeze(1))  
            output = F.log_softmax(output, dim=1) 
            decoder_outputs.append(output)
            attentions.append(attn_weights)

            if target_tensor is not None and random.random() < teacher_forcing_ratio:
                if i < target_tensor.size(1):
                    decoder_input = target_tensor[:, i].unsqueeze(1)  
                else:
                    decoder_input = torch.full((batch_size, 1), EOS_token, dtype=torch.long, device=encoder_outputs.device)  
            else:
                top1 = output.argmax(1)  
                decoder_input = top1.unsqueeze(1)  

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)

        return decoder_outputs, (hidden, cell), attentions
