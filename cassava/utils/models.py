from torch import nn
import timm

class CassavaNet(nn.Module):
    def __init__(self, num_classes, encoder='tf_efficientnet_b5_ns', dropout=0.01):
        super().__init__()
        self.dropout = dropout
        self.backbone = timm.create_model(encoder, pretrained=True)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.Linear(self.backbone.num_classes, num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_params(model, lr=3e-3, reduce=0.1):
    return  [
        {'params': model.backbone.parameters(), 'lr': lr * reduce},
        {'params': model.classifier.parameters(), 'lr': lr},
    ]
     