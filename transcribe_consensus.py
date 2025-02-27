import os
import glob
import difflib
import whisper

# Get environment variables for model and language
MODEL = os.environ.get("MODEL", "turbo")
LANGUAGE = os.environ.get("LANGUAGE", None)  # e.g., "English" (if not provided, auto-detect)

def align_two(seq1, seq2):
    """
    Aligns two token sequences and returns two lists of equal length,
    padding with None where tokens are missing.
    """
    sm = difflib.SequenceMatcher(None, seq1, seq2)
    aligned1 = []
    aligned2 = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            aligned1.extend(seq1[i1:i2])
            aligned2.extend(seq2[j1:j2])
        elif tag == 'replace':
            tokens1 = seq1[i1:i2]
            tokens2 = seq2[j1:j2]
            L = max(len(tokens1), len(tokens2))
            tokens1 += [None] * (L - len(tokens1))
            tokens2 += [None] * (L - len(tokens2))
            aligned1.extend(tokens1)
            aligned2.extend(tokens2)
        elif tag == 'delete':
            aligned1.extend(seq1[i1:i2])
            aligned2.extend([None] * (i2 - i1))
        elif tag == 'insert':
            aligned1.extend([None] * (j2 - j1))
            aligned2.extend(seq2[j1:j2])
    return aligned1, aligned2

def consensus_two(seq1, seq2):
    """
    Computes a column-wise consensus for two aligned token sequences.
    If one token is None, chooses the other. If both tokens differ,
    prints a debug message and defaults to the token from seq1.
    """
    aligned1, aligned2 = align_two(seq1, seq2)
    consensus = []
    for idx, (t1, t2) in enumerate(zip(aligned1, aligned2)):
        if t1 is None and t2 is None:
            continue  # Both gaps, skip.
        elif t1 is None:
            consensus.append(t2)
        elif t2 is None:
            consensus.append(t1)
        elif t1 == t2:
            consensus.append(t1)
        else:
            print(f"Column {idx} mismatch: '{t1}' vs '{t2}'; choosing '{t1}'")
            consensus.append(t1)
    return consensus

def consensus_three(seq1, seq2, seq3):
    """
    Computes a consensus among three token sequences by merging the first two,
    then merging that result with the third.
    """
    print("Merging first two transcripts...")
    cons12 = consensus_two(seq1, seq2)
    print("Merging consensus of first two with the third transcript...")
    final_consensus = consensus_two(cons12, seq3)
    return final_consensus

def process_audio_file(audio_file, model):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    final_filename = os.path.join("/audiofiles", f"{base_name}.txt")
    
    # Skip if transcript already exists
    if os.path.exists(final_filename):
        print(f"Transcript {final_filename} already exists. Skipping '{audio_file}'.")
        return

    print(f"\nProcessing file: {audio_file}")
    transcripts = []
    temp_files = []
    
    # Transcribe three times
    for i in range(1, 4):
        print(f"Transcribing attempt {i} for {audio_file}...")
        if LANGUAGE:
            result = model.transcribe(audio_file, language=LANGUAGE, beam_size=7, best_of=7)
        else:
            result = model.transcribe(audio_file)
        transcript = result["text"]
        transcripts.append(transcript)
        temp_filename = f"{base_name}_{i}.txt"
        with open(temp_filename, "w") as f:
            f.write(transcript)
        temp_files.append(temp_filename)
    
    # Tokenize transcripts into words
    token_lists = [t.split() for t in transcripts]
    
    print("Computing consensus transcription using sequence alignment...")
    consensus_tokens = consensus_three(token_lists[0], token_lists[1], token_lists[2])
    final_transcript = " ".join(consensus_tokens)
    
    with open(final_filename, "w") as f:
        f.write(final_transcript)
    print(f"Final consensus transcript saved to {final_filename}")
    
    # Remove temporary transcript files
    for fname in temp_files:
        try:
            os.remove(fname)
            print(f"Removed temporary file {fname}")
        except OSError as e:
            print(f"Error removing file {fname}: {e}")

def main():
    # Look for all .mp3 files in the current working directory
    audio_files = glob.glob("/audiofiles/*.mp3")
    if not audio_files:
        print("No .mp3 files found in /audiofiles.")
        return

    # Load the Whisper model once
    print(f"Loading Whisper model '{MODEL}' ...")
    model = whisper.load_model(MODEL)
    print("Model loaded.")

    for audio_file in audio_files:
        process_audio_file(audio_file, model)

if __name__ == "__main__":
    main()
