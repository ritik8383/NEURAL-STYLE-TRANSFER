import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import time
import functools

# --- Configuration ---
# Path to your content image and style image
CONTENT_PATH = 'content.jpg' # Replace with your content image file path
STYLE_PATH = 'style.jpg'   # Replace with your style image file path

# Output directory for styled images
OUTPUT_DIR = 'styled_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image dimensions (adjust for desired quality vs. speed)
# Larger dimensions mean higher quality but slower processing and more memory usage.
IMG_HEIGHT = 500
IMG_WIDTH = 500

# Weights for the loss functions (tune these for different results)
CONTENT_WEIGHT = 1e3
STYLE_WEIGHT = 1e-2
TOTAL_VARIATION_WEIGHT = 30 # To reduce noise

# Number of optimization steps/iterations
EPOCHS = 10
STEPS_PER_EPOCH = 100

# Layers to use for content and style extraction
# These are standard choices from the VGG19 network
CONTENT_LAYERS = ['block5_conv2'] # Usually a deeper layer for content
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1' # Multiple layers to capture style at different scales
]

# --- Helper Functions ---

def load_img(path_to_img):
    """Loads an image and preprocesses it for VGG19."""
    img = Image.open(path_to_img).convert('RGB')
    # Resize while maintaining aspect ratio, then crop or pad to desired size
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img) # VGG-specific preprocessing
    return tf.convert_to_tensor(img)

def deprocess_img(processed_img):
    """Converts a preprocessed image back to displayable format."""
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, "Input to deprocess_img must be a 3 or 4-dim array"

    # Perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1] # Convert BGR to RGB

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def display_images(content, style, generated=None):
    """Displays content, style, and generated images."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3 if generated is not None else 2, 1)
    plt.imshow(deprocess_img(content))
    plt.title('Content Image')
    plt.axis('off')

    plt.subplot(1, 3 if generated is not None else 2, 2)
    plt.imshow(deprocess_img(style))
    plt.title('Style Image')
    plt.axis('off')

    if generated is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(deprocess_img(generated))
        plt.title('Generated Image')
        plt.axis('off')
    plt.show()

# --- Model Definition ---

def get_vgg_model():
    """Builds a VGG19 model without the top classification layer."""
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False # Freeze VGG weights, we only use it as a feature extractor
    
    # Get outputs for specified content and style layers
    content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    
    # Create a new model that outputs features from these layers
    model_outputs = content_outputs + style_outputs
    return tf.keras.Model(vgg.input, model_outputs)

# --- Loss Functions ---

def content_loss(base_content, target_content):
    """Calculates content loss (Mean Squared Error)."""
    return tf.reduce_mean(tf.square(base_content - target_content))

def gram_matrix(input_tensor):
    """Calculates the Gram Matrix for a given feature map."""
    # Use einsum for efficient batch-wise dot product
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(base_style, target_style):
    """Calculates style loss based on Gram matrices."""
    gram_base = gram_matrix(base_style)
    gram_target = gram_matrix(target_style)
    return tf.reduce_mean(tf.square(gram_base - gram_target))

def total_variation_loss(img):
    """Calculates total variation loss to encourage smoothness."""
    x_var = tf.square(img[:, :, 1:, :] - img[:, :, :-1, :])
    y_var = tf.square(img[:, 1:, :, :] - img[:, :-1, :, :])
    return tf.reduce_sum(tf.pow(x_var, 0.5)) + tf.reduce_sum(tf.pow(y_var, 0.5))

class StyleContentModel(tf.keras.models.Model):
    """Combines VGG model with loss calculation."""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = get_vgg_model()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1] range."""
        inputs = inputs * 255.0 # Scale input to VGG expected range [0, 255]
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        style_features = [style_layer[0] for style_layer in style_outputs]
        content_features = [content_layer[0] for content_layer in content_outputs]
        
        return {'content': content_features, 'style': style_features}

# --- Main Style Transfer Loop ---

@tf.function()
def train_step(image, extractor, content_features, style_features, optimizer):
    with tf.GradientTape() as tape:
        model_outputs = extractor(image)
        
        # Calculate content loss
        content_loss_val = tf.add_n([
            content_loss(model_outputs['content'][i], content_features[i])
            for i in range(len(CONTENT_LAYERS))
        ])
        content_loss_val *= CONTENT_WEIGHT

        # Calculate style loss
        style_loss_val = tf.add_n([
            style_loss(model_outputs['style'][i], style_features[i])
            for i in range(len(STYLE_LAYERS))
        ])
        style_loss_val *= STYLE_WEIGHT

        # Calculate total variation loss
        tv_loss_val = total_variation_loss(image) * TOTAL_VARIATION_WEIGHT

        total_loss = content_loss_val + style_loss_val + tv_loss_val

    # Compute gradients and apply them
    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0)) # Ensure pixel values are in [0,1] range

    return total_loss, content_loss_val, style_loss_val, tv_loss_val

def run_style_transfer(content_path, style_path, num_iterations=1000):
    """Executes the neural style transfer process."""
    # Load and prepare images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Convert to float32 [0,1] range for optimization
    content_image = tf.image.convert_image_dtype(deprocess_img(content_image), tf.float32)
    style_image = tf.image.convert_image_dtype(deprocess_img(style_image), tf.float32)

    # Initialize the generated image with the content image
    # We use tf.Variable so its pixels can be optimized
    generated_image = tf.Variable(content_image)

    # Display initial images
    print("Content and Style Images:")
    display_images(content_image, style_image)

    # Build the style content extractor model
    extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)

    # Get target features from content and style images
    content_features = extractor(content_image)['content']
    style_features = extractor(style_image)['style']

    # Optimizer (Adam is a good choice for this task)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    print("\nStarting Style Transfer...")
    start_time = time.time()
    for i in range(num_iterations):
        loss_t, content_l, style_l, tv_l = train_step(
            generated_image, extractor, content_features, style_features, optimizer
        )
        if i % 50 == 0:
            print(f"Iteration {i}: Total Loss: {loss_t.numpy():.2f}, "
                  f"Content Loss: {content_l.numpy():.2f}, "
                  f"Style Loss: {style_l.numpy():.2f}, "
                  f"TV Loss: {tv_l.numpy():.2f}")

    end_time = time.time()
    print(f"\nStyle transfer finished in {end_time - start_time:.2f} seconds.")

    # Save and display the final generated image
    final_image_array = deprocess_img(generated_image.numpy())
    final_image = Image.fromarray(final_image_array)
    output_filename = os.path.join(OUTPUT_DIR, f'styled_image_{int(time.time())}.jpg')
    final_image.save(output_filename)
    print(f"Styled image saved to: {output_filename}")

    print("\nFinal Result:")
    display_images(content_image, style_image, generated_image.numpy())


if __name__ == '__main__':
    # You need to replace 'content.jpg' and 'style.jpg' with your actual image paths.
    # For example:
    # content_path = 'path/to/your/content_photo.jpg'
    # style_path = 'path/to/your/style_painting.jpg'

    # Download example images if they don't exist
    if not os.path.exists(CONTENT_PATH):
        tf.keras.utils.get_file(CONTENT_PATH, 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    if not os.path.exists(STYLE_PATH):
        tf.keras.utils.get_file(STYLE_PATH, 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    run_style_transfer(CONTENT_PATH, STYLE_PATH, num_iterations=1000)
