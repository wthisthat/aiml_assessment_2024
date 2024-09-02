# AI/ML Take Home Assessment 2024

## Overview
In this repo, two distinct project app are stored in respective folder (refer challenge title below). Both shares the same requirements.txt due to specific folder structure required for Streamlit hosting & easier local development environment handling.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/wthisthat/aiml_assessment_2024.git

2. Navigate to the project directory:
   ```bash
   cd aiml_assessment_2024

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Note: The PyTorch libraries (torch, torchaudio, torchvision) are CPU-compiled in the requirements.txt. If you wanted to run the applications with GPU for faster processing time for Traffic Analysis, replace the versions accordingly in requirements.txt before installing.
   ```
    torch==2.4.0+cu121
    torchaudio==2.4.0+cu121
    torchvision==0.19.0+cu121
   ```
   
## Challenge 1: Traffic Analysis (folder `traffic analysis`)
Sample videos for inference: https://drive.google.com/drive/folders/1Lk2DWxpQ6Pf2nNzRCJtWZ3Ig9BewALGS?usp=sharing

To run in local environment, follow steps below to setup:
1. Ensure the requirements.txt are installed, take note on the CPU/GPU compiled PyTorch libraries.
2. Run the application (Ensure current terminal directory is at `aiml_assessment_2024/`)
   ```bash
   streamlit run .\traffic_analysis\app.py
3. Once the Streamlit page is showed in browser, you are ready to go to either upload video or enable webcam input stream.
4. To stop the execution, ensure the browser tab is opened and enter "Ctrl + C" in the terminal.
5. To replace the YOLOv8 custom weight file, store the `.pt` file under `traffic_analysis` folder. Then update the pathing in `app.py` line 9.
   ```bash
   model = YOLO('traffic_analysis/{CUSTOM WEIGHT FILENAME.pt}')
   ```

## Challenge 3: Generative AI (folder `rag_chatbot`)
Demo web app: https://aimlassessment-chatbot.streamlit.app/

To run in local environment, follow steps below to setup:
1. Replace the API key placeholder with your personal API key in .env accordingly
2. Run the application (Ensure current terminal directory is at `aiml_assessment_2024/`)
   ```bash
   streamlit run .\rag_chatbot\app.py
3. Once the Streamlit page is showed in browser, you are ready to go and chat with the bot!
4. To stop the execution, ensure the browser tab is opened and enter "Ctrl + C" in the terminal.

