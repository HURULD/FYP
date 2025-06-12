#Experiment 1/2 large room
python .\src\main.py .\room_large_low_reverb.yaml -v room_render -p pgf
Copy-Item output.pgf room_large.pgf

# Experiment 3/4 small room
python .\src\main.py .\room_small_high_reverb.yaml -v room_render -p pgf
Copy-Item output.pgf room_small.pgf

# Experiment 5/6 large room with noise
python .\src\main.py .\room_large_low_reverb_noise.yaml -v room_render -p pgf
Copy-Item output.pgf room_large_noise.pgf

# Experiment 7/8 small room with noise
python .\src\main.py .\room_small_high_reverb_noise.yaml -v room_render -p pgf
Copy-Item output.pgf room_small_noise.pgf