import pstats
from pstats import SortKey

def analyze_performance(profile_file):
    """Return the cumulative time taken from the AdaptiveFilters.py:33(full_simulate) call
    """
    return profile_file.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('AdaptiveFilters.py')

if __name__ == "__main__":
    import os
    import re
    # Find all files of form "profile_*" in current dir
    profile_files = [f for f in os.listdir('.') if re.match(r'profile_\d+', f)]
    profile_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))  # Sort by number in filename
    # Every 5th profile file is a different experiment, exclude these
    profile_files_filter = [pf for i, pf in enumerate(profile_files) if i+1 % 5 != 0]
    for pf in profile_files_filter[0:4]: # First 4 experiments for now
        print(f"Analyzing performance for {pf}:")
        analyze_performance(pstats.Stats(pf))
        print("\n" + "="*50 + "\n")
    
    # Sepearate analysis for 5th profiles
    print("Analyzing performance for profile_5:")
    pstats.Stats("profile_5").strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)