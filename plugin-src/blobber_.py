'''
Blobber is a generic spot array analyzer
It is compatible with stacks
copyright Artur Yakimovich 2019

'''
from __future__ import with_statement, division
from ij import IJ
from ij.plugin.frame import RoiManager
from ij.plugin.filter import Analyzer
from ij.gui import GenericDialog, NonBlockingGenericDialog, Roi
from ij.measure import ResultsTable


def createArrayCoord(hdx, hdy, vdx, vdy, nCols, nRows, x0, y0):
	x = [[] for j in range(nRows)]
	y = [[] for j in range(nRows)]
	for j in range(nRows):
		for i in range(nCols):
			#x[j].append('{},{}'.format(i,j))
			x[j].append(int(i*hdx+j*vdx + x0))
			#y[j].append('{},{}'.format(i,j))
			y[j].append(int(i*hdy+j*vdy + y0))
	return x, y

def makeSelections(x,y,rm,imp,circRad):
	rm.reset()
	for j in range(len(x)):
		for i in range(len(x[-1])):
			imp.setRoi(IJ.OvalRoi(x[j][i],y[j][i],circRad*2,circRad*2))
			rm.addRoi(imp.getRoi())
	rm.runCommand(imp,"Show All with labels");
	gd = NonBlockingGenericDialog('Confirm ROI positions')
	gd.addMessage('Are the ROIs positioned correctly? Move them if necessary')
	gd.showDialog()
	if gd.wasCanceled():
		return None
	return imp, rm

def measureSelections(imp, rm, nCols, nRows):
	rt = ResultsTable.getResultsTable()
	rt.reset()
	IJ.run("Set Measurements...", "area mean min centroid integrated stack redirect=None decimal=3")
	for i in range(nCols):
		for j in range(nRows):
			rm.select(i+j)
			for s in range(1,imp.getNSlices()+1):
				print('analyzing slice {}'.format(s))
				imp.setSlice(s);
				#rm.runCommand(imp,"Measure")
				IJ.run(imp, "Measure", "");
	rm.reset()

	rt.updateResults()

	return imp, rm

def runGUI():

	defaultRad = 20
	defaultNCols = 20
	defaultNRows = 5
	
	gd = GenericDialog("Blobber - v0.1 alpha")
	gd.addMessage("\tBlobber is a generic spot array analyzer \n\t\t\t\t\t\t\t\t\t\t\t\t\t(copyright Artur Yakimovich 2019)\n\n")

	gd.addNumericField("Number_of_columns", defaultNCols, 0)
	gd.addNumericField("Number_of_rows", defaultNRows, 0)
	gd.addNumericField("Circle_radius_(px)", defaultRad, 0)
	nCols = int(gd.getNextNumber())
	nRows = int(gd.getNextNumber())
	rad = int(gd.getNextNumber())
	gd.showDialog()
	if gd.wasCanceled():
		return None
	runScript(nCols, nRows, rad)

def getEdgeCoord(x,y,circRad,imp,msg):
	IJ.run(imp, "Select None", "");
	imp.setRoi(IJ.OvalRoi(x,y,circRad*2,circRad*2))
	gd = NonBlockingGenericDialog(msg)
	gd.addMessage('Select '+msg+' coordinates')
	gd.showDialog()
	if gd.wasCanceled():
		return None
	# wait for user input
	coord = imp.getRoi().getContourCentroid()

	IJ.run(imp, "Select None", "");
	return coord[0]-circRad,coord[1]-circRad
	

def runScript(nCols, nRows, circRad):
	print('___Running Blobber___')
	
	circXUpLeft = 33.
	circYUpLeft = 915.
	circXUpRight = 1033.
	circYUpRight = 627.
	circXLowLeft = 95.
	circYLowLeft = 1125.


	imp = IJ.getImage()
	nSlices = imp.getNSlices()
	print('The image has {} slices'.format(nSlices))
	
	circXUpLeft, circYUpLeft = getEdgeCoord(circXUpLeft, circYUpLeft, circRad, imp, 'Upper Left')
	print('upper left x,y: {},{}'.format(circXUpLeft, circYUpLeft))
	circXUpRight, circYUpRight = getEdgeCoord(circXUpRight, circYUpRight,circRad,imp, 'Upper Right')
	print('upper right x,y: {},{}'.format(circXUpRight, circYUpRight))
	circXLowLeft, circYLowLeft = getEdgeCoord(circXLowLeft, circYLowLeft,circRad,imp, 'Lower Left')
	print('lower left x,y: {},{}'.format(circXLowLeft, circYLowLeft))
	
	

	
	hOffsetY = (circYUpRight - circYUpLeft) / (nCols - 1)
	hOffsetX = (circXUpRight - circXUpLeft) / (nCols - 1)
	vOffsetY = (circYLowLeft - circYUpLeft) / (nRows - 1)
	vOffsetX = (circXLowLeft - circXUpLeft) / (nRows - 1)
	
	
	x, y = createArrayCoord(hOffsetX, hOffsetY, vOffsetX, vOffsetY, nCols, nRows, circXUpLeft, circYUpLeft)
	RM = RoiManager()
	rm = RM.getRoiManager()
	imp, rm = makeSelections(x,y,rm,imp,circRad)
	imp, rm = measureSelections(imp, rm, nCols, nRows)
	IJ.run(imp, "Select None", "");
	#rm.multiMeasure(imp).show('Blober results')
	#rm.reset()


			
	#rm.runCommand(imp,"Delete");
if __name__ in ['__builtin__','__main__']:
	runGUI()

