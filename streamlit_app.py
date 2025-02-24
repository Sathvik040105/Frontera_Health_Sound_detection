import streamlit as st
import os
import tempfile
from inference import AudioInference

class CFG:
    sample_rate = 16000
    yamnet_frame_duration = 0.96
    yamnet_hop_duration = 0.48
    wav2vec_duration = 10.0
    yamnet_duration = 10.0
    num_classes = 3
    int2label = {
        0: 'infant_cry',
        1: 'scream',
        2: 'normal_utterance'
    }
    yamnet_samples_per_frame = int(yamnet_frame_duration * sample_rate)
    yamnet_hop_samples = int(yamnet_hop_duration * sample_rate)
    n_mels = 64
    window_size = int(0.025 * sample_rate)
    hop_length = int(0.01 * sample_rate)
    fmin = 125
    fmax = 7500

def main():
    st.title("Audio Classification App")
    st.write("Upload an audio file to classify it as infant cry, scream, or normal utterance")

    # Initialize the model
    cfg = CFG()
    try:
        inferencer = AudioInference(cfg)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Make prediction
            with st.spinner('Processing audio...'):
                result = inferencer.predict(tmp_path)

            # Display results
            st.subheader("Prediction Results")
            st.write(f"Predicted Class: {result['predicted_class']}")

            # Display probabilities
            st.subheader("Confidence Scores")
            for model_name, probs in result['probabilities'].items():
                st.write(f"\n{model_name.capitalize()} Model:")
                for i, prob in enumerate(probs):
                    class_name = cfg.int2label[i]
                    st.progress(float(prob))
                    st.write(f"{class_name}: {prob:.4f}")

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
