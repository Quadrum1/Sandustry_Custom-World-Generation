import numpy as np
import opensimplex
import matplotlib.pyplot as plt

from PIL import Image


def map_hex_to_rgb(_hex):
    new = [(_hex >> 16) & 0xFF, (_hex >> 8) & 0xFF, _hex & 0xFF]
    return np.array(new)


materials = {
    "FLUX": map_hex_to_rgb(0xaf00e0),
    "DIRT": map_hex_to_rgb(0x000000),
    "WATER": map_hex_to_rgb(0x9966ff),
    "AIR": map_hex_to_rgb(0xffffff),
    "CAVE": map_hex_to_rgb(0x993300),
    "GRASS": map_hex_to_rgb(0x00ff00),
    "SPORES": map_hex_to_rgb(0x00e000),
    "ICE": map_hex_to_rgb(0x66ccff),
    "BEDROCK": map_hex_to_rgb(0xaaaaaa),
}


# Frequency in [0,1], reasonably as 1/n
# Higher n leads to bigger puddles, but also less of them
# Smaller n leads to the inverse.
def run_2dnoise(x, y, freq):
    return opensimplex.noise2(x=x * freq, y=y * freq)


# Seeds the array with some noise expression.
def run_noise_step(shape, darray, material, threshold=0.0, freq=1 / 100, seed=None):
    if not seed:
        opensimplex.random_seed()
    else:
        opensimplex.seed(seed)

    for iy, ix in np.ndindex(shape):
        noise = run_2dnoise(ix, iy, freq)

        if noise > threshold:
            darray[iy, ix] = material

    return darray


# Generates a static world border to keep in all the fluids and sand.
def world_border(array):
    width = 10

    for ix in range(0, width):
        array[:, ix] = materials["BEDROCK"]
        array[:, -ix] = materials["BEDROCK"]

    for iy in range(0, width):
        array[-width:, :] = materials["BEDROCK"]

    return array


def underground_generation():
    # Default is just everything covered in dirt.
    # Add soil
    dims = (700, 1280)
    array = np.full((dims[0], dims[1], 3), materials["DIRT"])

    # Add water in another step
    array = run_noise_step(dims, array, materials["WATER"], threshold=0.5, freq=1 / 200)

    # Add flux in another step
    array = run_noise_step(dims, array, materials["FLUX"], threshold=0.7, freq=1 / 75)

    # Some spores
    array = run_noise_step(dims, array, materials["SPORES"], threshold=0.6, freq=1 / 60)

    array = world_border(array)
    return array


def terrain(array, dims):
    maximum_height = 50
    surface_heights = []

    # Wavey Hills
    for ix in range(dims[1]):
        height_var = run_2dnoise(ix, 0, freq=1 / 100)
        start_height = dims[0] - int(height_var * maximum_height) - 100
        surface_heights.append(start_height)

        array[start_height:(start_height + 20), ix] = materials["GRASS"]
        array[(start_height + 20):, ix] = materials["DIRT"]

    # Seed some water near the surface
    # Never breach the surface
    for iy, ix in np.ndindex(dims):
        if iy < surface_heights[ix] + 10: continue

        noise = run_2dnoise(ix, iy, 1 / 100)

        if noise > 0.25:
            array[iy, ix] = materials["WATER"]

    return array


def mountains(array, dims):
    x_max = dims[1]
    y_max = dims[0]
    if x_max < 200:  # Map should we big enough, otherwise this shouldn't be used
        return array

    maximum_height = 50
    width = 200

    # Scale the gradient of mountain growth using parameters above
    gradient = int(width / maximum_height)

    for ix in range(0, width):
        height_var_left = run_2dnoise(ix, 0, freq=1 / 10)
        height_var_right = run_2dnoise(ix, y_max, freq=1 / 10)

        floor_height_left = gradient * ix + int(height_var_left * maximum_height)
        floor_height_right = gradient * ix + int(height_var_right * maximum_height)

        array[floor_height_left:(floor_height_left + 10), ix] = materials["ICE"]
        array[(floor_height_left + 10):, ix] = materials["BEDROCK"]

        array[floor_height_right:(floor_height_right + 10), x_max - ix - 1] = materials["ICE"]
        array[(floor_height_right + 10):, x_max - ix - 1] = materials["BEDROCK"]

    return array


def overworld_generation(seed=None):
    if not seed:
        opensimplex.random_seed()
    else:
        opensimplex.seed(seed)

    dims = (580, 1280)
    array = np.full((dims[0], dims[1], 3), materials["AIR"])

    array = terrain(array, dims)
    array = mountains(array, dims)

    return array


overworld = overworld_generation()
underground = underground_generation()
out = np.concatenate((overworld, underground), axis=0)
out = out.astype(np.uint8)

# This is only for Plot visualisation in PyCharm and other IDEs
plt.imshow(out, cmap='viridis', interpolation="nearest")  # You can use any colormap
plt.show()

# Save to disk.
image = Image.fromarray(out, 'RGB')
image.save('map_blueprint_playtest.png')
