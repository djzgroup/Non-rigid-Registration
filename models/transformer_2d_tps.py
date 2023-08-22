from utils.util import *
import geotnf.point_tnf
import geotnf.transformation


class get_model(nn.Module):
    def __init__(self, d_model=512, channel=3):
        super(get_model, self).__init__()
        self.transformer = nn.Transformer(d_model, num_encoder_layers=4, num_decoder_layers=4)
        # self.stn=STNkd(d_model)
        self.pointnet = PointNetEncoder(channel, d_model)
        self.pointnetDecoder = PointNetDecoder(d_model)
        self.fc1 = nn.Linear(d_model*91, d_model)
        self.fc2 = nn.Linear(d_model, d_model//2)
        self.fc3 = nn.Linear(d_model//2, 18)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):
        # temp1, temp2 = Z_order_sorting.apply(encoder_input[:, :, :3], encoder_input[:, :, 3:6])
        # encoder_input = temp1
        # temp1, temp2 = Z_order_sorting.apply(decoder_input[:, :, :3], decoder_input[:, :, 3:6])
        # decoder_input = temp1
        encoder_input, decoder_input = encoder_input.permute(0, 2, 1), decoder_input.permute(0, 2, 1)
        embed_input = self.pointnet(encoder_input)
        embed_output = self.pointnet(decoder_input)
        embed_input, embed_output = embed_input.permute(2, 0, 1), embed_output.permute(2, 0, 1)  # [N, B, d_model]
        transformer_out = self.transformer(embed_input, embed_output)  # [N, B ,d_model]
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, N, d_model]
        # stn_out=self.stn(transformer_out.permute(0,2,1)) #[B,d_model,d_model]
        # warpped_feat=transformer_out@stn_out   #[B, N, d_model]
        warpped_feat = transformer_out  # [B, N, d_model]
        bs=encoder_input.shape[0]
        warpped_feat=torch.flatten(warpped_feat,1)  #[N , N * d_model]
        out1=torch.nn.functional.leaky_relu(self.fc1(warpped_feat))
        out2=torch.nn.functional.leaky_relu(self.fc2(out1))
        out3=self.fc3(out2)

        theta = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        theta = Variable(torch.from_numpy(theta).float()).cuda().unsqueeze(0).repeat(bs, 1)
        out3=out3+theta
        a = geotnf.point_tnf.PointTnf(use_cuda=True)
        decoder_out = a.tpsPointTnf(out3, decoder_input[:,0:2,])

        # displacement = self.pointnetDecoder(warpped_feat.permute(0, 2, 1))  # [B,3,N]
        # warped = displacement + decoder_input[:, 0:3, :]
        loss1 = chamfer_loss(encoder_input[:, :2, :], decoder_out, ps=decoder_out.size()[-1])

        # # task2 To train the decoder
        # resumed_decoder_input = self.pointnetDecoder(embed_output.permute(1, 2, 0))
        # loss2 = PC_distance(resumed_decoder_input, decoder_input[:, :3, :])

        return decoder_out.permute(0, 2, 1), loss1,loss1


if __name__ == '__main__':
    model = get_model(128, 6)
    a = torch.rand((2, 1024, 6))
    b = torch.rand((2, 1024, 6))
    loss = model(a, b)
    print(loss)
