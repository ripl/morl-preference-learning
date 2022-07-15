#!/bin/bash
xvfb-run -s "-screen 0 1400x900x24" # virtual screen for video recording
python3 -m PbMORL.main # launch main program