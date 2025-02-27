import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Download a sample image from the web
url = 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg'
resp = urllib.request.urlopen(url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 100, 200)

# Display results using matplotlib
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(edges, cmap='gray')
plt.title("Edges Detected")
plt.axis("off")

plt.show()
