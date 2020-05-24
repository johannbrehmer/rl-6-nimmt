import torch
from torch import nn


class SechsNimmtStateNormalization(nn.Module):
    def __init__(self, cards=104, rows=4, action=False):
        super().__init__()
        self.cards = cards
        self.rows = rows
        self.action = action

    def forward(self, input):
        flatten = False
        if len(input.size()) == 1:
            input = input.unsqueeze(0)
            flatten = True

        pos = 0
        outputs = []

        # Action
        if self.action:
            outputs.append(self._normalize(input[:, pos:pos+1], min_=0, max_=self.cards - 1))
            pos += 1

        # Number of players
        outputs.append(self._normalize(input[:, pos:pos+1], min_=0, max_=6))
        pos += 1

        # Cards per row
        outputs.append(self._normalize(input[:, pos:pos+self.rows], min_=1, max_=5))
        pos += self.rows

        # Highest card per row
        outputs.append( self._normalize(input[:, pos:pos+self.rows], min_=0, max_=self.cards - 1))
        pos += self.rows

        # Score per row
        outputs.append(self._normalize(input[:, pos:pos+self.rows], min_=1, max_=10))
        pos += self.rows

        # Raw game state
        outputs.append(self._normalize(input[:, pos:], min_=-1, max_=self.cards - 1))

        outputs = torch.cat(outputs, axis=1)
        if flatten:
            outputs = outputs.squeeze()

        return outputs


    @staticmethod
    def _normalize(input, min_, max_, out_min=-1., out_max=1.):
        return out_min + (out_max - out_min) * (input - min_) / (max_ - min_)
