# Create a temporary file with corrected content
with open("visualization/traj_plot.py", "r") as f:
    content = f.read()
# Replace the constant definition with properly formatted version
    content = content.replace("# Define a constant for layer separation in the single sceneLAYER_SEPARATION_OFFSET = 10.0 # Moderate layer separation distance", "# Define a constant for layer separation in the single scene\nLAYER_SEPARATION_OFFSET = 10.0 # Moderate layer separation distance")
with open("visualization/traj_plot.py", "w") as f:
    f.write(content)
print("Fixed LAYER_SEPARATION_OFFSET formatting in traj_plot.py")
