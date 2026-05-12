Vision-Clair
Vision-Clair is a real-time, hybrid Edge/Cloud AI system designed to bridge the communication gap between individuals who use American Sign Language (ASL) and those who do not.

Using a standard webcam, the system reads ASL fingerspelling frame-by-frame, buffers the letters into words, corrects grammar using Large Language Models, and vocalizes the final sentence using high-fidelity Text-to-Speech.

Built By
Gheorghe Chirica

Alexandra Ciobanu

Features
Real-Time Edge ML: Uses a custom Convolutional Neural Network (CNN) trained on the Sign Language MNIST dataset to detect static ASL letters (A-Y, excluding motion-based J and Z).

Smart Buffering: Implements a frame-holding queue to prevent accidental keystrokes and "debounces" camera inputs for accurate spelling.

Geometric System Controls: Uses lightning-fast, rule-based MediaPipe hand geometry to trigger system actions without straining the ML model.

AI Grammar Correction: Passes raw, fingerspelled text through Google Gemini 2.5 Flash to fix typos, add punctuation, and humanize the sentence.

Natural Vocalization: Integrates ElevenLabs API to speak the polished sentence out loud instantly.

Tech Stack
Computer Vision: OpenCV, Google MediaPipe (Tasks API)

Machine Learning: TensorFlow, Keras, Pandas, NumPy

Cloud AI / APIs: Google Generative AI (Gemini), ElevenLabs (TTS)

Environment: Python 3.11, python-dotenv

Getting Started
1. Prerequisites
You must have Python 3.11 installed on your system.
You will also need API keys from:

Google AI Studio (Gemini)

ElevenLabs

Create a .env file in the root directory and add your keys:

Fragment de cod
GEMINI_API_KEY=your_gemini_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL
2. Installation
Clone the repository and set up a virtual environment:

Bash
# Clone the repository
git clone https://github.com/GOUT648/vision-clair.git
cd vision-clair

# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. Training the Brain (One-Time Setup)
Before running the camera, you must train the neural network on the provided MNIST dataset to generate the model.h5 file.

Bash
python train.py
Note: This will take 1-3 minutes depending on your CPU. It will automatically save the model to your folder upon reaching 15 epochs.

4. Running Vision-Clair
Once the model is trained, launch the live camera translation engine:

Bash
python main.py
How to Use (Gesture Controls)
Hold your hand up to the webcam. The system will automatically crop your hand and begin reading ASL letters.

To control the flow of text, use the following built-in geometric gestures:

Type Letters (A-Y): Hold the ASL sign steady.

Space (End Word): Hold up an Open Palm (all four fingers extended).

Delete (Backspace): Give a Thumbs Down gesture.

Speak Sentence: Hold up a Peace Sign (V). This triggers the AI to polish your built sentence and speak it aloud.

Press q on your keyboard while the window is active to quit the application.

Project Structure
Plaintext
vision-clair/
├── .env                    # API Keys (Not tracked by Git)
├── .gitignore              # Ignores large files and virtual envs
├── index.html              # Marketing Landing Page
├── main.py                 # Core application and camera loop
├── requirements.txt        # Python dependencies
├── sign_mnist_test.csv     # Kaggle Dataset (Testing)
├── sign_mnist_train.csv    # Kaggle Dataset (Training)
└── train.py                # TensorFlow CNN builder
