from utils.util import *


class _get_model(nn.Module):
    def __init__(self, d_model=128, channel=3,npoint=None):
        super(_get_model, self).__init__()
        self.transformer = nn.Transformer(d_model, num_encoder_layers=4, num_decoder_layers=4)
        # self.stn=STNkd(d_model)
        # self.pointnet = PointNetEncoder(channel, d_model)
        self.pointnet=PointNetSetAbstractionMsg(npoint,[0.1,0.2,0.4],[32,64,128],channel-3,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.reduce=nn.Conv1d(320,d_model,1)
        self.bn=nn.BatchNorm1d(d_model)
        self.pointnetDecoder = DispGenerator(d_model)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):
        # temp1  = Z_order_sorting.apply(encoder_input[:, :, :3])
        # encoder_input = temp1
        # temp1  = Z_order_sorting.apply(decoder_input[:, :, :3])
        # decoder_input = temp1
        encoder_input, decoder_input = encoder_input.permute(0, 2, 1), decoder_input.permute(0, 2, 1)   #[B C N]

        _,embed_input = self.pointnet(encoder_input,None)
        embed_input=F.relu(self.bn(self.reduce(embed_input)))

        _,embed_output = self.pointnet(decoder_input,None)
        embed_output=F.relu(self.bn(self.reduce(embed_output)))

        embed_input, embed_output = embed_input.permute(2, 0, 1), embed_output.permute(2, 0, 1)  # [N, B, d_model]
        transformer_out = self.transformer(embed_input, embed_output)  # [N, B ,d_model]
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, N, d_model]
        warpped_feat = transformer_out

        displacement = self.pointnetDecoder(warpped_feat.permute(0, 2, 1))  # [B,3,N]
        warped = displacement + decoder_input[:, 0:3, :]
        loss1 = chamfer_loss(encoder_input[:, :3, :], warped, ps=warped.size()[-1])

        return warped.permute(0, 2, 1), loss1 #, loss2

class get_model(nn.Module):
    def __init__(self, d_model=128, channel=3,npoint=None):
        super(get_model, self).__init__()
        self.model=_get_model(d_model,channel,npoint)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):
        B,N,_=encoder_input.size()
        warped1,loss1 = self.model(encoder_input, decoder_input)
        warped2,loss2=self.model(decoder_input,warped1)
        cycle_loss = torch.mean(torch.sqrt(torch.sum((encoder_input[...,0:3]-warped2) ** 2, dim=2)), dim=-1).mean()
        return warped1,loss1,cycle_loss*0.005



if __name__ == '__main__':
    model = _get_model(128, 6)
    a = torch.rand((2, 1024, 6))
    b = torch.rand((2, 1024, 6))
    loss = model(a, b)
    print(loss)
