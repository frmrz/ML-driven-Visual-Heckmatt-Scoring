import pydicom
import matplotlib.pyplot as plt

def load_dicom(file_path):
    """Load DICOM file and return dataset."""
    return pydicom.dcmread(file_path)

def extract_all_info(dicom_dataset):
    """Extract all available information from the DICOM dataset."""
    info = {}
    for elem in dicom_dataset:
        if elem.VR != "SQ":  # Exclude Sequences
            field_name = elem.name.replace(" ", "_")
            info[field_name] = str(elem.value)
    return info

def visualize_image(dicom_dataset):
    """Visualize the DICOM image."""
    plt.imshow(dicom_dataset.pixel_array, cmap=plt.cm.gray)
    plt.title("DICOM Image")
    plt.show()

def visualize_metadata(metadata):
    """Visualize metadata in a simple text format."""
    for key, value in metadata.items():
        print(f"{key}: {value}")

def save_image_as_png(dicom_dataset, output_file_path):
    """Save the DICOM image data as a PNG file."""
    plt.imsave(output_file_path, dicom_dataset.pixel_array, cmap=plt.cm.gray)

def convert_dicom_to_png(input_dir, output_dir):
    """Recursively read all DICOM images in a directory and its subdirectories, and save them as PNG images in an output directory."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                dicom_dataset = pydicom.dcmread(file_path)
                output_file_path = os.path.join(output_dir, file.replace(".dcm", ".png"))
                plt.imsave(output_file_path, dicom_dataset.pixel_array, cmap=plt.cm.gray)

# Example Usage
file_path = '/media/francesco/DEV001/PROJECT-FSHD/DATA/dicomData_FSHD/6/anon_6_Biceps brachii_L_1.dcm'
dicom_dataset = load_dicom(file_path)
metadata = extract_all_info(dicom_dataset)

visualize_metadata(metadata)
visualize_image(dicom_dataset)

input_dir = "/media/francesco/DEV001/PROJECT-FSHD/DATA/dicomData_FSHD/"
output_dir = "/media/francesco/DEV001/PROJECT-FSHD/DATA/converted_dicom"
convert_dicom_to_png(input_dir, output_dir)

# use plotly to visualize the dicom image
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def count_transitions(image_path):
    # Load the png image
    image = Image.open(image_path)
    image = np.array(image)

    # Take the last column of the image in only one channel
    image = image[:, :, 0]
    last_column = image[:, -1] > 100

    # Find differences between adjacent elements
    transitions = (last_column[:-1] == True) & (last_column[1:] == False)
    count = np.sum(transitions)

    # measure distance between first and second transition
    first_transition = np.where(transitions)[0][0]
    second_transition = np.where(transitions)[0][1]
    distance = second_transition - first_transition
    cf = 1 / distance

    return count, distance, cf

# now make a loop that counts the transitions for all the images in the dataset and stores the results in a dataframe   
import os
import pandas as pd

root = '/media/francesco/DEV001/PROJECT-FSHD/DATA/converted_dicom'
df = pd.DataFrame(columns=['image', 'transitions', 'distance', 'cf'])

for file in tqdm(os.listdir(root)):
    image_path = os.path.join(root, file)
    transitions, distance, cf = count_transitions(image_path)
    df.loc[len(df)] = {'image': file, 'transitions': transitions, 'distance': distance, 'cf': cf}

df.to_csv('/media/francesco/DEV001/PROJECT-FSHD/DATA/TABULAR/conversion_factor.csv', index=False)
df.to_excel('/media/francesco/DEV001/PROJECT-FSHD/DATA/TABULAR/conversion_factor.xlsx', index=False)

# plot image with plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import numpy as np
from PIL import Image
import os

image_path = '/media/francesco/DEV001/PROJECT-FSHD/DATA/converted_dicom/anon_2320_Depressor anguli oris_L_3.png'
image = Image.open(image_path)
image = np.array(image)

fig = px.imshow(image)
fig.show()
