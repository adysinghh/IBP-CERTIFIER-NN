# visualize.py
import matplotlib.pyplot as plt

def show_certified_image(x, certified):
    plt.imshow(x.squeeze().numpy(), cmap='gray')
    plt.title("Certified" if certified else "Not Certified")
    plt.axis('off')
    plt.show()
