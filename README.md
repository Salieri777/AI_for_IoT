# SER-ESC-50
pytorch - DSP - audio classification

</br></br>

## Sonic Environment Recognition

A simple audio classifier that recieves a wav file and returns a classification from 50 categories

</br></br>

## How to Use

    git clone https://github.com/FinchMF/SER-ESC-50.git

    pip install -r requirements.txt

    bash run.sh

    python app.py



Once all requirements are installed, executing **run.sh** will:

*   fetch all necessary data (ESC-50 audio dataset)
*   train both a CNN and a modified resNet 34 

</br></br>

Once the models are trained, executing **app.py** will:

*   run a local server with a simplistic front end to interact with the classifier 
*   upload wav file and the classifier will respond with a classificaiton 

</br></br>

## Coming Soon
Producionized application using AWS





