import torchaudio
import torch
import glob

DATASET_PATH = "../dataset/"

INSTRUMENTS = [
    "Bansuri",
    "Shehnai",
    "Santoor",
    "Sarod",
    "Sitar"
]

CLASS_LABELS = {
    "Bansuri": 0,
    "Shehnai": 1,
    "Santoor": 2,
    "Sarod"  : 3,
    "Sitar"  : 4    
}

NEW_SAMPLE_RATE = 8000
WINDOW_SIZE = 20000


# REQUIRES THE INSTRUMENT LEVEL FOLDERS TO BE CREATED PRIOR TO EXECUTION
def main():
    i = 0
    for instrument in INSTRUMENTS:
        instrument_dataset_path = f"{DATASET_PATH}{instrument}/"
        mp3_files = glob.glob(instrument_dataset_path + "*.mp3")
        output_tensors_list = []
        print(instrument)
        for song_filename in mp3_files:
            print(song_filename)
            waveform, sample_rate = torchaudio.load(song_filename)
            waveform = torchaudio.transforms.Resample(sample_rate, NEW_SAMPLE_RATE)(waveform)
            windows_list = torch.split(waveform.unsqueeze(0), WINDOW_SIZE, dim=2)
            output_tensor = torch.vstack(windows_list[:-1])
            output_tensors_list.append(output_tensor)
        instrument_specific_tensor = torch.vstack(output_tensors_list)
        del output_tensors_list,
        tensor_list = torch.split(instrument_specific_tensor, 16, 0)
        for i, x in enumerate(tensor_list):
            torch.save(x.clone(), f"{instrument}/{instrument}_{i+1}.pt")


if __name__ == '__main__':
    main()
