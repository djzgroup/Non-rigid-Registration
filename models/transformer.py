from utils.util import *

class get_model(nn.Module):
    def __init__(self, d_model=128, channel=3,npoint=None):
        super(get_model, self).__init__()
        self.transformer = nn.Transformer(d_model, num_encoder_layers=4, num_decoder_layers=4)
        # self.stn=STNkd(d_model)
        self.pointnet = PointNetEncoder(channel, d_model)
        self.pointnetDecoder = DispGenerator(d_model)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,**kwargs):
        # temp1  = Z_order_sorting.apply(encoder_input[:, :, :3])
        # encoder_input = temp1
        # temp1  = Z_order_sorting.apply(decoder_input[:, :, :3])
        # decoder_input = temp1
        encoder_input, decoder_input = encoder_input.permute(0, 2, 1), decoder_input.permute(0, 2, 1)
        embed_input = self.pointnet(encoder_input)
        embed_output = self.pointnet(decoder_input)
        embed_input, embed_output = embed_input.permute(2, 0, 1), embed_output.permute(2, 0, 1)  # [N, B, d_model]
        transformer_out = self.transformer(embed_input, embed_output)  # [N, B ,d_model]
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, N, d_model]
        # stn_out=self.stn(transformer_out.permute(0,2,1)) #[B,d_model,d_model]
        # warpped_feat=transformer_out@stn_out   #[B, N, d_model]
        warpped_feat = transformer_out

        displacement = self.pointnetDecoder(warpped_feat.permute(0, 2, 1))  # [B,3,N]
        warped = displacement + decoder_input[:, 0:3, :]
        loss1 = chamfer_loss(encoder_input[:, :3, :], warped, ps=warped.size()[-1])

        # task2 To train the decoder
        # resumed_decoder_input = self.pointnetDecoder(embed_output.permute(1, 2, 0))
        # loss2 = PC_distance(resumed_decoder_input, decoder_input[:, :3, :])

        return warped.permute(0, 2, 1), loss1 #, loss2


if __name__ == '__main__':
    model = get_model(128, 6)
    a = torch.rand((2, 1024, 6))
    b = torch.rand((2, 1024, 6))
    loss = model(a, b)
    print(loss)
