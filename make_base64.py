import base64

with open("sample_drawings.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

print("Length:", len(encoded))

with open("test_base64.txt", "w") as f:
    f.write(encoded)

print("Saved to test_base64.txt")