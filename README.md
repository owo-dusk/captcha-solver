# captcha-solver
Solve Image based captchas for OwObot

OwObot (https://owobot.com) is a Discord Bot, which provides a competitive text based game. This project aims at solving text based Image captchas in order to automate this text based game.
We do this with the help of YOLO and Labelme.


`train.py` -> To actually train the model. No pre-processing of images was done (was lazy). Perhaps you should consider preprocessing, as that may fetch better results.
`test_onnx.p` -> After exporting to an ONNX model, we can test the captcha solver with this file. 
`examples` folder contains 5 example captchas from OwO-Bot

All captcha images in use in my dataset was fetched with the help of https://github.com/Tyrrrz/DiscordChatExporter
https://github.com/wkentaro/labelme was used to annotate captcha images.
There may be better alternatives to Labelme available (Like Roboflow for example), I choose Labelme to keep everything local.

Took around 2~ days to get this to 90~ % accuracy with only 500~ images used for training. With better control of images in use, perhaps it could be reduced to 300 as I managed to achieve80% accuracy with just 300.

Feel free to use this mode **in compliance with GNU GPL V3** licence :>
Overall this was a fun project, I could learn alot from it!
