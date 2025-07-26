import os
import glob
import re
import imageio.v2 as imageio

def create_heat_animation(output_dir='heat_evolution', fps=10, output_name='heat_evolution.gif'):
    """
    Create an animation from the saved heat equation solution images
    
    Args:
        output_dir: Directory containing the saved images
        fps: Frames per second for the animation
        output_name: Name of the output GIF file
    """
    # Get all image files and sort by epoch number
    image_files = glob.glob(os.path.join(output_dir, 'heat_epoch_*.png'))
    image_files.sort(key=lambda x: int(re.search(r'heat_epoch_(\d+).png', x).group(1)))
    
    # Read images and create animation
    images = []
    for image_file in image_files:
        images.append(imageio.imread(image_file))
    
    # Save as GIF
    output_gif = os.path.join(output_dir, output_name)
    imageio.mimsave(output_gif, images, fps=fps)
    print(f"Heat equation animation saved to {output_gif}")

if __name__ == "__main__":
    # Create the animation with default parameters
    create_heat_animation(fps=5)