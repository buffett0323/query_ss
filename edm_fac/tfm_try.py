from audiotools import AudioSignal
from audiotools import transforms as tfm
import random
import soundfile as sf
import audiotools
print("Audiotools", audiotools.__file__)
print("Transform", tfm.__file__)


process = tfm.Compose(
    [
        tfm.ClippingDistortion(perc=("uniform", 0.0, 0.5)),
        # tfm.MuLawQuantization(),
        # tfm.LowPass(prob=0.5),
    ],
    name="process",
    prob=1.0,
)

process2 = tfm.Compose(
    [
        tfm.SeqPerturbReverse(
            # method="fixed",
            num_segments=5,
            fixed_second=0.3,
            reverse_prob=("uniform", 0.5, 1),
        ),
        tfm.PitchShift(
            n_semitones=("choice", [-2, -1, 0, 1, 2]),
            quick=True,
        )
    ]
)


path = "/home/buffett/dataset/EDM_FAC_DATA/beatport/train/001bd429-3113-485d-84b2-dfdf56ae9132_chorus_other_0.wav"

signal, _ = sf.read(
    path,
    start=0,
    frames=44100*5,
)

ad = AudioSignal(signal, 44100)
rn = random.uniform(-2, 2)
ad.pitch_shift(rn)
ad.write('sample_audio/orig.wav')


# kwargs = process2.instantiate(state=0)
# output = process2(ad.clone(), **kwargs)

# output.write('sample_audio/proc.wav')
