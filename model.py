import torch
import torch.nn as nn


# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Attention, self).__init__()
#         self.attn = nn.Linear(hidden_dim, 1)

#     def forward(self, lstm_output):
#         # lstm_output: [batch, seq_len, hidden_dim]
#         scores = self.attn(lstm_output).squeeze(-1)  # [batch, seq_len]
#         weights = F.softmax(scores, dim=1)           # [batch, seq_len]
#         weighted_output = torch.bmm(weights.unsqueeze(1), lstm_output)  # [batch, 1, hidden_dim]
#         return weighted_output.squeeze(1)            # [batch, hidden_dim]
    
    
class TwoChannelLSTMClassifier(nn.Module):
    def __init__(self, input_channels=2, hidden_size=64, num_layers=2, num_classes=1, dropout_rate=0.3):
        super(TwoChannelLSTMClassifier, self).__init__()

        # Simple CNN architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),

            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),

            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # Fixed output size
            nn.Dropout(0.4)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, channels, sequence]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    #     self.features = nn.Sequential(
    #         nn.Conv1d(input_channels, 8, kernel_size=11, padding=5),
    #         nn.BatchNorm1d(8),
    #         nn.ReLU(),
    #         nn.MaxPool1d(8),
    #         nn.Dropout(0.2),
            
    #         nn.Conv1d(8, 16, kernel_size=7, padding=3),
    #         nn.BatchNorm1d(16),
    #         nn.ReLU(),
    #         nn.AdaptiveAvgPool1d(2),
    #         nn.Dropout(0.2)
    #     )
        
    #     self.classifier = nn.Sequential(
    #         nn.Linear(16 * 2, 8),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(8, num_classes)
    #     )
    
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x

    # # Slightly more capacity than MinimalEEGClassifier
    #     self.features = nn.Sequential(
    #         nn.Conv1d(input_channels, 12, kernel_size=11, padding=5),
    #         nn.BatchNorm1d(12),
    #         nn.ReLU(),
    #         nn.MaxPool1d(8),
    #         nn.Dropout(0.2),
            
    #         nn.Conv1d(12, 24, kernel_size=7, padding=3),
    #         nn.BatchNorm1d(24),
    #         nn.ReLU(),
    #         nn.MaxPool1d(4),
    #         nn.Dropout(0.2),
            
    #         nn.Conv1d(24, 32, kernel_size=5, padding=2),
    #         nn.BatchNorm1d(32),
    #         nn.ReLU(),
    #         nn.AdaptiveAvgPool1d(3),
    #         nn.Dropout(0.25)
    #     )
        
    #     self.classifier = nn.Sequential(
    #         nn.Linear(32 * 3, 16),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(16, num_classes)
    #     )
    
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x



    #     self.lstm_hidden_size = hidden_size
    #     self.lstm_num_layers = num_layers

    #     # CNN feature extractor
    #     self.cnn_features = nn.Sequential(
    #         # First conv block
    #         nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
    #         nn.BatchNorm1d(32),
    #         nn.ReLU(),
    #         nn.MaxPool1d(4),
    #         nn.Dropout(0.3),

    #         # Second conv block
    #         nn.Conv1d(32, 64, kernel_size=5, padding=2),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.MaxPool1d(4),
    #         nn.Dropout(0.3),

    #         # Third conv block
    #         nn.Conv1d(64, 128, kernel_size=3, padding=1),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(),
    #         nn.Dropout(0.4)
    #     )
        
    #     # LSTM layer
    #     self.lstm = nn.LSTM(
    #         input_size=128,  # CNN output channels
    #         hidden_size=hidden_size,
    #         num_layers=num_layers,
    #         batch_first=True,
    #         dropout=0.3 if num_layers > 1 else 0,
    #         bidirectional=True  # Use bidirectional LSTM for better temporal modeling
    #     )
        
    #     # Final classifier
    #     self.classifier = nn.Sequential(
    #         nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
    #         nn.ReLU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(64, num_classes)
    #     )

    # def forward(self, x):
    #     # x shape: [batch, channels, sequence]
    #     batch_size = x.size(0)
        
    #     # CNN feature extraction
    #     cnn_out = self.cnn_features(x)  # [batch, 128, sequence_length]
        
    #     # Prepare for LSTM: transpose to [batch, sequence, features]
    #     lstm_input = cnn_out.transpose(1, 2)  # [batch, sequence_length, 128]
        
    #     # LSTM processing
    #     lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
    #     # Use the last output for classification
    #     # For bidirectional LSTM, take the last output
    #     final_output = lstm_out[:, -1, :]  # [batch, lstm_hidden_size * 2]
        
    #     # Classification
    #     output = self.classifier(final_output)
        
    #     return output