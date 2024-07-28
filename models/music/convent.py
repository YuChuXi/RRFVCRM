from midiutil import MIDIFile

# Dictionary to map base values to notes
base_to_note = {
    'A': 60,  # C4
    'T': 62,  # D4
    'C': 64,  # E4
    'G': 65   # F4
}

def convert_dna_to_midi(dna_sequence, output_file):
    # Create MIDIFile object with one track
    midi = MIDIFile(1)

    # Set track name and tempo
    track = 0
    midi.addTrackName(track, 0, "DNA to MIDI")
    midi.addTempo(track, 0, 120)

    # Iterate over DNA sequence and add notes to MIDI
    time = 0
    for base in dna_sequence:
        if base in base_to_note:
            note = base_to_note[base]
            midi.addNote(track, 0, note, time, 1, 100)
        time += 1

    # Write MIDI data to file
    with open(output_file, "wb") as file:
        midi.writeFile(file)

# Example usage
dna_sequence = "ATCGATCGATCG"
output_file = "output.mid"
convert_dna_to_midi(dna_sequence, output_file)
print(f"MIDI file '{output_file}' created successfully!")