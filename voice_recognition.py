#installing speaker verification toolkit which read the audio files
pip install speaker-verification-toolkit
#installing numba which converts the audio into number arrays, 0.48 is a version of numba
pip install numba==0.48
#importing speaker_verification_toolkit as svt 
import speaker_verification_toolkit.tools as svt
#importing librosa 
import librosa
#testing the librosa file
librosa.load('example.wav')

# reading the 2 audio file  
data1, sr = librosa.load('vm1.wav', sr=16000, mono=True)
data2, sr = librosa.load('vm2.wav', sr=16000, mono=True)

data1 = svt.rms_silence_filter(data1)
data2 = svt.rms_silence_filter(data2)

#MFCC technique aims to develop the features from the audio signal
data1 = svt.extract_mfcc(data1)
data2 = svt.extract_mfcc(data2)

#by comparing two audio files it gives the some numerical value
print(
    'The difference between voice1 and voice2 is',
    svt.compute_distance(data1,data2)
)

#reads the third audio
data3, sr = librosa.load('example.wav', sr=16000, mono=True)
data3 = svt.rms_silence_filter(data3)
data3 = svt.extract_mfcc(data3)

#give an numerical value for both the audio
svt.compute_distance(data3, data2)

#now it compates the audio 3 and 1 with 2 and say which audio intensity is nearest to audio 2, either 3 or 1 .
svt.find_nearest_voice_data([ data3, data1], data2)
