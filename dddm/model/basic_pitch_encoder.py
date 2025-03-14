from model.basic_pitch_torch.model import BasicPitchTorch
import torch



def init_basic_pitch_model():
    pt_model = BasicPitchTorch()
    pt_model.load_state_dict(torch.load('model/assets/basic_pitch_pytorch_icassp_2022.pth'))
    pt_model.eval()
    return pt_model


def basic_pitch_encoder(input_array, model):
    model.eval()
    with torch.no_grad():
        output_pt = model(input_array)
        contour_pt, note_pt, onset_pt = output_pt['contour'], output_pt['note'], output_pt['onset']
    return contour_pt, note_pt, onset_pt



if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    pt_model = init_basic_pitch_model()
    pt_model.to(device)
    y_torch = torch.randn(8, 1, 44100*8).to(device)
    
    contour_pt, note_pt, onset_pt = basic_pitch_encoder(y_torch, pt_model)

    print(contour_pt.shape)
    print(note_pt.shape)
    print(onset_pt.shape)