import torch.nn as nn
import torch.nn.functional as F

from models.modules.transformation import TPS_SpatialTransformerNetwork
from models.modules.feature_extraction import ResNet_FeatureExtractor
from models.modules.sequence_modeling import BidirectionalLSTM

class STARNET(nn.Module):

    def __init__(self, imgH, imgW, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(STARNET, self).__init__()
        
        input_channel = nc
        output_channel = 512
        hidden_size = nh
        class_size = nclass
        
        self.Transformation = TPS_SpatialTransformerNetwork(F = 20, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num = input_channel)
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        self.SequenceModeling_withCorrLSTM = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, class_size))
        
        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, class_size))

        
    def forward(self, input, finetune=False):
            
        input = self.Transformation(input)

        visual_feature = self.FeatureExtraction(input)
        b, c, h, w = visual_feature.size()
        #print(h)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(3, 0, 1, 2))  # [b, c, h, w] -> [w, b, c, h]
        #print(visual_feature.size())
        visual_feature = visual_feature.squeeze(3)

        rnn_feature = self.SequenceModeling(visual_feature)
        if finetune == True:
            rnn_feature = self.SequenceModeling_withCorrLSTM(visual_feature)

        output = F.log_softmax(rnn_feature, dim=2)
        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero

   
