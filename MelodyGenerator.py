import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import music21 as m21
from pre import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:

    def __init__(self,model_path="Model/model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH,"r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self,seed,num_steps,max_sequence_length, temperature):
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-max_sequence_length:]
            onehot_seed = keras.utils.to_categorical(seed,num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            if output_symbol == "/":
                break
            melody.append(output_symbol)
        return melody



    def _sample_with_temperature(self,probabilities,temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions)/np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self,melody,step_duration=0.25,format="midi",file_name="Melody_file/mel.mid"):
        stream = m21.stream.Stream()
        start_symbol = None
        step = 1
        for i,symbol in enumerate(melody):
            if symbol != "_" or i+1 ==len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration*step
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarter_Length=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarter_Length=quarter_length_duration)
                    stream.append(m21_event)
                    step = 1
                start_symbol = symbol
            else:
                step +=1
        stream.write(format,file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed ="55 _ _ _ 64 _ "
    melody = mg.generate_melody(seed , 500,SEQUENCE_LENGTH,0.3)
    print(melody)
    mg.save_melody(melody)



