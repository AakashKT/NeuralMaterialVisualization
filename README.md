<h1 align="center">A Flexible Neural Renderer for Material Visualization</h1>
<p align="center"><b>SIGGRAPH Asia 2019, Technical Briefs (In review)</b></p>
<div align="center">
  <span>
    <a href="https://scholar.google.co.in/citations?user=itJ7vawAAAAJ&hl=en">Aakash KT<sup>1</sup></a>,
    <a href="https://scholar.google.co.in/citations?user=h1_Uc2QAAAAJ&hl=en">Parikshit Sakurikar<sup>1</sup></a>,
    <a href="https://scholar.google.co.in/citations?hl=en&user=OSZDITwAAAAJ">Saurabh Saini<sup>1</sup></a>,
    <a href="https://scholar.google.co.in/citations?user=3HKjt_IAAAAJ&hl=en">P J Narayanan<sup>1</sup></a>
  </span>
</div>
<p align="center"><sup>1</sup>CVIT, IIIT Hyderabad</p>
<hr>
<img src="https://aakashkt.github.io/teaser.jpg" width="900px" height="319px">
<div align="center">
  <span>
    <a href="https://aakashkt.github.io/neural-renderer-material-visualization.html">[Project page]</a>
    <a href="">[arXiv]</a>
    <a href="https://drive.google.com/drive/folders/1DXcVPr-g7H5SefmrOSs3xRGdMof0SBwZ?usp=sharing">[Data]</a>
    <a href="https://www.youtube.com/embed/yiBGF6Jycck">[Video]</a>
    <a href="">[bibtex]</a>
  </span>
</div>
<hr>
<p><b>Abstract</b><br>
  Photo realism in computer generated imagery is crucially dependent on how well an artist is able to recreate real-world materials in the scene. The workflow for material modeling and editing typically involves manual tweaking of material parameters and uses a standard path tracing engine for visual feedback. A lot of time may be spent in iterative selection and rendering of materials at an appropriate quality. In this work, we propose a convolutional neural network based workflow which quickly generates high-quality ray traced material visualizations on a shaderball. Our novel architecture allows for control over environment lighting and assists material selection along with the ability to render spatially-varying materials. Additionally, our network enables control over environment lighting which gives an artist more freedom and provides better visualization of the rendered material. Comparison with state-of-the-art denoising and neural rendering techniques suggests that our neural renderer performs faster and better. We provide a interactive visualization tool and release our training dataset to foster further research in this area.
</p>
<p><b>Acknowledgements</b><br>
  We thank all the reviewers of SIGGRAPH Asia 2019, for their valuable comments and suggestions.
</p>

# Running the code
## Prerequisites
This code was tested on UBuntu 18.04, with Python 2.7. <br>

<p><b>Python 2.7 dependencies:</b><br>
tensorflow-1.3.1<br>
torch-1.1.0 (pytorch)<br>
torchvision-0.0.2.post3<br>
flask-1.0.2<br>
pillow-5.1.0<br>
numpy-1.13.3<br>
opencv-python-4.1.0.25<br>
</p>

## Steps to run
<ul>
  <li>Clone this repo (git clone https://github.com/AakashKT/NeuralMaterialVisualization/).</li>
  <li>Download Blender 2.8 from https://www.blender.org/download/ . Extract the contents and place them inside the 'blender-2.80' directory.</li>
  <li>Download the network weights from: https://drive.google.com/drive/folders/1DXcVPr-g7H5SefmrOSs3xRGdMof0SBwZ?usp=sharing and place the '.pt' file inside this directory.</li>
  <li>Run the interactive tool with: python main.py .</li>
  <li>The tool will be available at localhost:8000 .</li>
</ul>

## Training Data
The training dataset is available at https://drive.google.com/drive/folders/1DXcVPr-g7H5SefmrOSs3xRGdMof0SBwZ?usp=sharing
