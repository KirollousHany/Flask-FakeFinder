from flask import Flask, request, jsonify, url_for, send_file
from TTS.api import TTS
import os
from scipy.io import wavfile
import librosa
import noisereduce as nr
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
#cors = CORS(app, resources={r"/api/": {"origins": ""}})

def convert_voices(source_wav, target_wav):
    # Initialize TTS model
    tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False)
    
    # Perform voice conversion
    output_voice = tts.voice_conversion(source_wav=source_wav, target_wav=target_wav)
    
    # Save the output voice to a temporary file
    with open("output_voice.wav", "wb") as output_file:
        sampling_rate = 22050  # Default sampling rate
        wavfile.write(output_file, sampling_rate, (output_voice * 32767.0).astype('int16'))
    
    audio_url = url_for('get_denoised_audio', filename="output_voice.wav")  # Generate URL
    
    return audio_url

@app.route('/voice_conversion', methods=['POST'])
def voice_conversion():
    # Check if the POST request contains source and target voice files
    if 'source_voice' not in request.files or 'target_voice' not in request.files:
        return jsonify({'error': 'Source and target voice files are required.'}), 400
    
    source_wav = request.files['source_voice']
    target_wav = request.files['target_voice']
    
    # Save source and target voice files
    source_wav_path = "source_voice.wav"
    target_wav_path = "target_voice.wav"
    source_wav.save(source_wav_path)
    target_wav.save(target_wav_path)
    
    # Perform voice conversion
    output_voice_url = convert_voices(source_wav_path, target_wav_path)
    
    # Remove source and target voice files
    os.remove(source_wav_path)
    os.remove(target_wav_path)
    
    # Return success message with audio URL
    return jsonify({
        'message': 'Voice conversion successful',
        'output_voice_url': output_voice_url
    })

def convert_voices_new(text, speaker_wav, language):
    # Initialize TTS model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts(text=text, speaker_wav=speaker_wav, language=language)

    # Run TTS
    output_voice_path = tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language)
    
    # Read the generated WAV file to obtain the sampling rate and audio data
    sampling_rate, output_voice = wavfile.read(output_voice_path)
    
    # Save the output voice to a temporary file
    with open("output_voice.wav", "wb") as output_file:
        wavfile.write(output_file, sampling_rate, output_voice)
    
    audio_url = url_for('get_denoised_audio', filename="output_voice.wav")  # Generate URL
    
    return audio_url

@app.route('/voice_conversion_new', methods=['POST'])
def voice_conversion_new():
    # Check if the POST request contains source voice file and text
    if 'source_voice' not in request.files or 'text' not in request.form or 'language' not in request.form:
        return jsonify({'error': 'Source voice file, text, and language are required.'}), 400
    
    source_wav = request.files['source_voice']
    text = request.form['text']
    language = request.form['language']
    
    # Save source voice file
    source_wav_path = "source_voice.wav"
    source_wav.save(source_wav_path)
    
    # Perform voice conversion
    output_voice_url = convert_voices_new(text, source_wav_path, language)
    
    # Remove source voice file
    os.remove(source_wav_path)
    
    # Return success message with audio URL
    return jsonify({
        'message': 'Voice conversion successful',
        'output_voice_url': output_voice_url
    })

    
@app.route('/reduce_noise', methods=['POST'])
def reduce_noise_api():
  """
  API endpoint to reduce noise from uploaded audio file.
  """
  # Check if audio file is present in request
  if 'audio_file' not in request.files:
    return jsonify({'error': 'No audio file uploaded'}), 555

  # Get the audio file
  audio_file = request.files['audio_file']

  # Read the audio data
  try:
    rate, data = wavfile.read(audio_file)
  except Exception as e:
    return jsonify({'error': f'Error reading audio file: {str(e)}'}), 400

  # Perform noise reduction
  try:
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
  except Exception as e:
    return jsonify({'error': f'Error during noise reduction: {str(e)}'}), 500

  # Save denoised audio to temporary file
  with open("denoised_audio.wav", "wb") as output_file:
    wavfile.write(output_file, rate, reduced_noise)
    audio_url = url_for('get_denoised_audio', filename="denoised_audio.wav")  # Generate URL

  # Return success message with audio data information and URL
  return jsonify({
      'message': 'Noise reduction successful',
      'original_rate': rate,
      'original_data_shape': data.shape,
      'denoised_audio_url': audio_url
  })

@app.route('/get_denoised_audio/<filename>')
def get_denoised_audio(filename):
  """
  Serves the denoised audio file based on the filename.
  """
  try:
    # Open the temporary file
    with open(filename, 'rb') as audio_file:
      audio_data = audio_file.read()
    # Set response headers for audio content
    return audio_data, 200, {'Content-Type': 'audio/wav'}
  except FileNotFoundError:
    return jsonify({'error': 'Denoised audio file not found'}), 404
