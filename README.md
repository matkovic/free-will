# free-will
AI project - Disprove the existence of the free will

My attempt for the "Free will experiment" using Emotiv EPOC EEG headset and OpenViBE platform (the idea for the experiment can be seen in this youtube video - https://www.youtube.com/watch?v=lmI7NnMqwLQ ). This project could have been done better with additional filters to get rid of noise/blinks etc. and more precise EEG data.

For receiving data from the EPOC headset, I used hemokit (https://github.com/nh2/hemokit) with following command: 
hemokit-dump.exe --format sensorbytes --serve 127.0.0.1:1337 --realtime

OpenViBE collects this data with acquisition server as described in https://github.com/nh2/hemokit/wiki/OpenVibe .

OpenViBE scenarios that I used are located in the openvibe-scenarios folder:
* free-will-bci-1-acquisition.xml - saving EEG data to CSV file, saving keyboard button presses into CSV file
* free-will-bci-2-classifier-trainer.xml - scenario with just a python script (see scripts/python-learning.py)

