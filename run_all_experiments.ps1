#Experiment 1 with all filters and covariance whitening on the large low reverb room
python .\src\main.py .\room_large_low_reverb.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E1
python .\src\main.py .\room_large_low_reverb.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E1
python .\src\main.py .\room_large_low_reverb.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E1
python .\src\main.py .\room_large_low_reverb.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E1
python .\src\main.py .\room_large_low_reverb.yaml -e rtf_accuracy_covariance_whitening_identity -o .\Experiments\E1

# Experiment 2 with all filters and covariance whitening on the large high reverb room
python .\src\main.py .\room_large_high_reverb.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E2
python .\src\main.py .\room_large_high_reverb.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E2
python .\src\main.py .\room_large_high_reverb.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E2
python .\src\main.py .\room_large_high_reverb.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E2
python .\src\main.py .\room_large_high_reverb.yaml -e rtf_accuracy_covariance_whitening_identity -o .\Experiments\E2

# Experiment 3 with all filters and covariance whitening on the small low reverb room
python .\src\main.py .\room_small_low_reverb.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E3
python .\src\main.py .\room_small_low_reverb.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E3
python .\src\main.py .\room_small_low_reverb.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E3
python .\src\main.py .\room_small_low_reverb.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E3
python .\src\main.py .\room_small_low_reverb.yaml -e rtf_accuracy_covariance_whitening_identity -o .\Experiments\E3

# Experiment 4 with all filters and covariance whitening on the small high reverb room
python .\src\main.py .\room_small_high_reverb.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E4
python .\src\main.py .\room_small_high_reverb.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E4
python .\src\main.py .\room_small_high_reverb.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E4
python .\src\main.py .\room_small_high_reverb.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E4
python .\src\main.py .\room_small_high_reverb.yaml -e rtf_accuracy_covariance_whitening_identity -o .\Experiments\E4

# Experiment 5 with all filters and covariance whitening on the large low reverb room with extra noise source
python .\src\main.py .\room_large_low_reverb_noise.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E5
python .\src\main.py .\room_large_low_reverb_noise.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E5
python .\src\main.py .\room_large_low_reverb_noise.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E5
python .\src\main.py .\room_large_low_reverb_noise.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E5
python .\src\main.py .\room_large_low_reverb_noise.yaml -e rtf_accuracy_covariance_whitening_noise -o .\Experiments\E5 --noise_length 1.3

# Experiment 6 with all filters and covariance whitening on the large high reverb room with extra noise source
python .\src\main.py .\room_large_high_reverb_noise.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E6
python .\src\main.py .\room_large_high_reverb_noise.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E6
python .\src\main.py .\room_large_high_reverb_noise.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E6
python .\src\main.py .\room_large_high_reverb_noise.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E6
python .\src\main.py .\room_large_high_reverb_noise.yaml -e rtf_accuracy_covariance_whitening_noise -o .\Experiments\E6 --noise_length 1.3

# Experiment 7 with all filters and covariance whitening on the small low reverb room with extra noise source
python .\src\main.py .\room_small_low_reverb_noise.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E7
python .\src\main.py .\room_small_low_reverb_noise.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E7
python .\src\main.py .\room_small_low_reverb_noise.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E7
python .\src\main.py .\room_small_low_reverb_noise.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E7
python .\src\main.py .\room_small_low_reverb_noise.yaml -e rtf_accuracy_covariance_whitening_noise -o .\Experiments\E7 --noise_length 1.3

# Experiment 8 with all filters and covariance whitening on the small high reverb room with extra noise source
python .\src\main.py .\room_small_high_reverb_noise.yaml -f LMS 2048 0.001 -e rtf_accuracy_filter -o .\Experiments\E8
python .\src\main.py .\room_small_high_reverb_noise.yaml -f NLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E8
python .\src\main.py .\room_small_high_reverb_noise.yaml -f PNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E8
python .\src\main.py .\room_small_high_reverb_noise.yaml -f IPNLMS 2048 0.1 -e rtf_accuracy_filter -o .\Experiments\E8
python .\src\main.py .\room_small_high_reverb_noise.yaml -e rtf_accuracy_covariance_whitening_noise -o .\Experiments\E8 --noise_length 1.3

# Run for learning curve plot
python .\src\main.py .\room_large_low_reverb.yaml -f IPNLMS 2048 0.1  -v rtf_learning_curve -p pgf -e rtf_accuracy_filter -o .\Experiments\E_learning_curve
Copy-Item output.pgf .\Experiments\E_learning_curve\learning_curve_IPNLMS_2048_0.1.pgf
# Run for rrir comparison plot
python .\src\main.py .\room_large_low_reverb.yaml -f IPNLMS 2048 0.1 -v rrir -p pgf -e rtf_accuracy_filter -o .\Experiments\E_rrir_compare
Copy-Item output.pgf .\Experiments\E_rrir_compare\rrir_IPNLMS_2048_0.1.pgf
