with open("visualization/dash_app.py", "r") as f:
    content = f.read()
content = content.replace("layer_separation=LAYER_SEPARATION_OFFSET", "layer_separation=50.0")
with open("visualization/dash_app.py", "w") as f:
    f.write(content)
print("Updated layer_separation in dash_app.py")
