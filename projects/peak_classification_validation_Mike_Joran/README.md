# Peak validation using the WFsim

Authors:
  * Mike Clark
  * Joran Angevaare
Based on previous work of Tianyu and Jelle combination of:
  * https://github.com/XENONnT/WFSim/blob/master/notebooks/bias_smearing_demo.ipynb
  * https://github.com/XENON1T/XeAnalysisScripts/tree/master/PeakFinderTest

**Perquisites**
  * https://github.com/XENONnT/straxen
  * https://github.com/AxFoundation/strax 
  * https://github.com/XENONnT/WFSim
  
**Usage**
  * The goal is to validate the peak classification by strax using the waveform simulator.
  * The main notebook is ``Peak_Classification_test.ipynb``
  This notebook compares the input of the waveform simulator to the output of the waveform simulator. This can be used as a probe to test how well strax is performing at reconstructing and especially classifying peaks. In order so we check if the input peaks called the ''truth'' is compared to the ''data''. This shows that some of the peaks are at the moment split incorrectly but most are found in ''straxen''.
  

**Known issues**
  * This tool is still under construction. Testing is still required.
  * There seems to be a 500 ns time difference between the WFsim truth and the WFsim result. This monkey patched at the moment.
  * Some of the ''dtypes'' in the ''truth'' are changed to comparable ''dtypes'' of the ''data''. 
