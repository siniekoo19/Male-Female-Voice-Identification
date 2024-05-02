# import librosa
# import numpy as np
# import pandas as pd
# from scipy.stats import skew

# def extract_acoustic_features(file_path, start, end):
#     data, sample_rate = librosa.load(file_path, sr=None, offset=start, duration=end-start)

#     mean_freq = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1))
#     sd = np.std(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1))
#     q25, q75 = np.percentile(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1), [25, 75])
#     iqr = q75 - q25
#     sp_ent = np.sum(librosa.feature.spectral_contrast(y=y, sr=sr))
#     sfm = np.mean(librosa.feature.spectral_flatness(y=y))


#     data = {
#         "sd": sd,
#         "IQR": iqr,
#         "sp.ent": sp_ent,
#         "sfm": sfm,
#         "meanfun": mean_freq
#     }

#     return data

# def voice_to_csv(input_file, output_file, start=0, end=None):
#     if end is None:
#         duration = librosa.get_duration(filename=input_file)
#         end = duration if duration < 20 else 20
    
#     # Extract features
#     features = extract_acoustic_features(input_file, start, end)
    
#     # Create DataFrame
#     df = pd.DataFrame([features])
    
#     # Save to CSV
#     # df.to_csv(output_file, index=False)
#     return df

# # Example usage
# # voice_to_csv("Data_amy.wav", "amy_acoustics.csv", start=0, end=20)

# # The feature names should match those that were passed during fit.


import librosa
import numpy as np
import pandas as pd

def extract_acoustic_features(file_path, start, end):
    y, sr = librosa.load(file_path, sr=None, offset=start, duration=end-start)

    # Calculate fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches)
    pitch_sd = np.std(pitches)
    
    # Calculate inter quantile range of fundamental frequency
    q25, q75 = np.percentile(pitches, [25, 75])
    pitch_iqr = q75 - q25
    
    # Calculate spectral entropy
    sp_ent = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Calculate spectral flatness
    sfm = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Calculate mean fundamental frequency
    meanfun = pitch_mean

    data = {
        "sd": pitch_sd,
        "IQR": pitch_iqr,
        "sp.ent": sp_ent,
        "sfm": sfm,
        "meanfun": meanfun
    }

    return data

def voice_to_csv(input_file, output_file, start=0, end=None):
    if end is None:
        duration = librosa.get_duration(filename=input_file)
        end = duration if duration < 20 else 20
    
    # Extract features
    features = extract_acoustic_features(input_file, start, end)
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Save to CSV
    # df.to_csv(output_file, index=False)
    return df
