# KMDPlotter.py
A KMDPlotter is a data visualization tool used for analysis of mass spectrometry. The goal is to remove irrelevant chemical and mechanical noise that the instrument gives as the result.

The program utilizes a drag and drop interface to upload excel files, which are made of 4 columns: mass, intensity, mass b and mass y. The masses are converted into Kendrick mass (KM) for x-axis and Kendrick mass defect (KMD) for y-axis. Mass scatters are displayed using blue points, while mass b are orange and y are green. 

The graph is made to be interactive for user-friendlyness. The interactive features consist of checkboxes, search engine, 'save data' button, as well as panning and zooming in. 

The search engine is made of a 'Search' and 'Delete' box. The data needs to be inputted in each box exactly as in the input file to be able to display it. The display contains it's KM and KMD values.

The checboxes toggle visibility of some graph elements: a line that connects the first and last scatter of each dataset, a scatter from each data set with maximal distance from that b or y line, parallel lines that go through those maximally distanced scatters, as well as the same line translated on the other side of the original b or y line, and scatters that lay within parallel lines on either side of b and y line. 

The scatter thats within the parallel lines is the relevant data, and we can count them and determine their percentage, which is also displayed on the graph. That is considered scatter within 'Limit 1'. However, theres a Limit 2, which includes scatters within distance of 0.085 from the original b and y lines. 

The scatter that lays within Limit 1 or 2 is defined by their KM, KMD and distance from the line. When 'Save New Scatter' button is pressed, a pop-up window appears allowing the user to save that data in a csv file.

This KMDPlotter.py repository contains 2 different code variations: KMD_2.py, KMD_1.py. KMD_2 does what has been explain so far, but the KMD_1, instead of having b and y lines, only has one ion line.

The other documents in the repo are a User Documentation for KMD_2.py, examples of input files for each python program and their output csv files.
