import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        weights = torch.tanh(self.attn(x))
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class CandlePatternAI(nn.Module):
    def __init__(self, input_size=15, seq_len=50, num_classes=3):
        super().__init__()
        
        # 1. Feature Extraction (CNN) - Added more regularization
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        
        # 2. Sequence Logic (LSTM) - Reduced hidden size to 64
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.2 if 1 > 1 else 0  # Only applies if num_layers > 1
        )
        
        # 3. Focus Logic (Attention)
        self.attention = Attention(64)
        
        # 4. Output classifier - Multi-layer with Dropout
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        cnn_in = x.permute(0, 2, 1)                      
        cnn_out = self.cnn(cnn_in)                        
        cnn_out = cnn_out.permute(0, 2, 1)                
        
        lstm_out, _ = self.lstm(cnn_out)                  
        
        attn_out = self.attention(lstm_out)               
        
        return self.classifier(attn_out)

if __name__ == "__main__":
    model = CandlePatternAI()
    test_input = torch.randn(8, 50, 15) 
    output = model(test_input)
    print(f"Test Output Shape: {output.shape}") 
