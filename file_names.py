import filecmp

# Path to the two directories you want to compare
dir1 = "/path/to/directory1"
dir2 = "/path/to/directory2"

# Compare the two directories
dirs_equal = filecmp.dircmp(dir1, dir2)

# Check if the two directories have the same files
if not dirs_equal.diff_files and not dirs_equal.left_only and not dirs_equal.right_only:
    print("The two directories have the same files.")
else:
    print("The two directories do not have the same files.")