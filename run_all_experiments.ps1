
# Experiment 1 with all filters and covariance whitening on the large low reverb room
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

