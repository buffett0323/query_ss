Cocochorales website:
https://magenta.tensorflow.org/datasets/cocochorales

Structure:
string_track000001
   |-- metadata.yaml
   |-- mix.mid
   |-- stems_MIDI
   |    |-- 0_violin.mid
   |    |-- 1_violin.mid
   |    |-- 2_viola.mid
   |    |-- 3_cello.mid 
   |-- mix.wav
   |-- stems_audio
        |-- 0_violin.wav
        |-- 1_violin.wav
        |-- 2_viola.wav
        |-- 3_cello.wav 

There are 28,179 samples of mixtures split into the train, validation, and test sets with a ratio of 70/20/10. The wave- forms are converted into mel spectrograms using 128 mel- filter bands, a window size of 1,024, and a hop length of 512. We crop a 320ms segment, or 10 spectral frames, from the sustain phase of each sample.