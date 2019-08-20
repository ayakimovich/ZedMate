'''
ZedMate v0.18 beta

ZedMate is a TrackMate based 2.5D prticle analyzer
copyright Artur Yakimovich 2018. MIT license.
Code is based on the discussion from http://forum.imagej.net/
http://forum.imagej.net/t/jython-macro-for-multi-channel-trackmate-analysis/7637
as well as Jean-Yves Tinevez original code:
https://gist.github.com/tinevez/7f32c6bbe45a1dd4e919209f8c053253

requires
TrackMate
TrackMate-extras

'''

import os
import fiji.plugin.trackmate.Spot as Spot
import fiji.plugin.trackmate.Model as Model
import fiji.plugin.trackmate.Settings as Settings
import fiji.plugin.trackmate.TrackMate as TrackMate
from fiji.plugin.trackmate import SelectionModel
from ij.plugin import HyperStackConverter
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer

import fiji.plugin.trackmate.detection.LogDetectorFactory as LogDetectorFactory

import fiji.plugin.trackmate.tracking.LAPUtils as LAPUtils
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking.oldlap import LAPTrackerFactory
import fiji.plugin.trackmate.extra.spotanalyzer.SpotMultiChannelIntensityAnalyzerFactory as SpotMultiChannelIntensityAnalyzerFactory

import ij. IJ as IJ
from ij.gui import GenericDialog, NonBlockingGenericDialog
import java.io.File as File
import java.util.ArrayList as ArrayList
from fiji.plugin.trackmate.io import TmXmlWriter
from ij import ImageStack, ImagePlus
from ij.process import FloatProcessor


def swapZT(imp):
	dims = imp.getDimensions() # default order: XYCZT
	IJ.log("Image dimensions are:")
	IJ.log(str(dims[0])+", "+str(dims[1])+", "+str(dims[2])+", "+str(dims[3])+", "+str(dims[4]))

	if (dims[4] == 1):
		IJ.log("Swapping dimensions...")
		imp.setDimensions(dims[2], dims[4], dims[3])
		dims = imp.getDimensions() # default order: XYCZT
		IJ.log("Current dimensions are:")
		IJ.log(str(dims[0])+", "+str(dims[1])+", "+str(dims[2])+", "+str(dims[3])+", "+str(dims[4]))
	return imp

def runBatch(inputDir, outputDir, extension, containString, targetChannel,dt, radius, threshold, \
				frameGap, linkingMax, closingMax, testMode, mimicryEmbd):
	for root, directories, filenames in os.walk(inputDir):
		for fileName in filenames:
			# Check for file extension
			if not fileName.endswith(extension):
				continue
			# Check for file name pattern
			if containString not in fileName:
				continue
			#imp = IJ.openImage("/Users/ayakimovich/Dropbox (LMCB)/Moona and Artur/IEVorCEV/SCR1_170713_siRNA_Alix_B5surface.lif - SCR_1.tif");
			imp = IJ.openImage(inputDir+fileName)
			IJ.log('Processing: '+inputDir+fileName)
			imp = swapZT(imp)
			#try:
			if(True):
				model, nChannels = runTrackMate(imp, targetChannel, dt, radius, threshold, frameGap, linkingMax, closingMax)
				imp.close()
				if testMode:
					selectionModel = SelectionModel(model)
					displayer =  HyperStackDisplayer(model, selectionModel, imp)
					displayer.render()
					displayer.refresh()
					return runGUI(targetChannel, dt, radius, threshold, frameGap, linkingMax, closingMax)
				
				outputFile = outputDir+fileName.replace(".tif", "")+"_ZedMate_output.csv"
				
				if mimicryEmbd:
					IJ.log('Saving mimicry embedding...')
					saveMimicryEmbedding(model,outputFile,nChannels)
					
				
				IJ.log('Writing ZedMate Measurements to: '+outputFile)
				writeMeasurements(model,outputFile,nChannels)
			#except TypeError:
			#	print("No particles detected in the current image, skipping to the next step")


def compileTracks(tm, nChannels, rowStr):
	tracksFormated = []
	trackIDs = tm.trackIDs(True)
	for trackID in trackIDs:
		tracks = tm.trackSpots(trackID)
	
		# Let's sort them by frame.
		trackList = ArrayList(tracks);
		#trackList.sort(Spot.nameComparator)
		
		for spot in trackList:
			values = [  spot.ID(), trackID, spot.getFeature('FRAME'), spot.getFeature('POSITION_X'), spot.getFeature('POSITION_Y') ]
			for i in range( nChannels ):
				values.append(spot.getFeature('MEAN_INTENSITY%02d' % (i+1)))					
			IJ.log(rowStr % tuple(values))
			tracksFormated.append(rowStr % tuple(values))
		
	tracksSorted = sorted(tracksFormated, key=lambda x: x[2],reverse=False)
	return tracksSorted


					
def runTrackMate(imp, targetChannel, dt, radius, threshold, frameGap, linkingMax, closingMax):
	# Get the number of channels 
	nChannels = imp.getNChannels()
	IJ.log("->Detection threshold used: "+str(threshold))
	IJ.log("->Number of frames is: "+str(imp.getStackSize()))
	IJ.log("->Target channel is: "+str(targetChannel))
	IJ.log( '->Number of channels to measure %d' % nChannels)
	# Setup settings for TrackMate
	settings = Settings()
	settings.setFrom( imp )
	settings.dt = dt

	# Spot analyzer: we want the multi-C intensity analyzer.
	settings.addSpotAnalyzerFactory( SpotMultiChannelIntensityAnalyzerFactory() )

	# Spot detector.
	settings.detectorFactory = LogDetectorFactory()
	settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
	settings.detectorSettings['RADIUS'] = radius
	settings.detectorSettings['THRESHOLD'] = threshold
	settings.detectorSettings['TARGET_CHANNEL'] = targetChannel

	# Spot tracker.
	#settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerFactory = LAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
	settings.trackerSettings['MAX_FRAME_GAP']  = frameGap
	settings.trackerSettings['LINKING_MAX_DISTANCE']  = linkingMax
	settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE']  = closingMax
	settings.trackerSettings['ALLOW_TRACK_MERGING'] = False

	settings.trackerSettings['ALLOW_GAP_CLOSING'] = False
	settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
	settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
	settings.trackerSettings['ALTERNATIVE_LINKING_COST_FACTOR'] = 0.5
	settings.trackerSettings['BLOCKING_VALUE'] = 1.0
	settings.trackerSettings['CUTOFF_PERCENTILE'] = 1.0
	
	#settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = 16.0
	# Run TrackMate and store data into Model.

	model = Model()
	trackmate = TrackMate(model, settings)



	if not trackmate.checkInput() or not trackmate.process():
		IJ.log('Could not execute TrackMate: ' + str( trackmate.getErrorMessage() ) )
	else:
		return model, nChannels

def saveMimicryEmbedding(model,outputFile,nChannels):
	tm = model.getTrackModel()
	trackIDs = tm.trackIDs(True)
	embSize = 14
	sliceNum = 0
	cutoff = 10
	embedding = ImageStack(embSize, embSize)
	
	
	for trackID in trackIDs:
		tracks = tm.trackSpots(trackID)
		# Let's sort them by frame.
		trackList = ArrayList(tracks)
		trackLength = len(trackList)
		if(trackLength <= cutoff):
			trackPic = FloatProcessor(nChannels, trackLength)
			trackPix = trackPic.getPixels()
			#print(len(trackPix))
			idxSpot = 0
			for spot in trackList:
				idxChan = 0		
				for iChan in range(nChannels):
						trackPix[idxSpot * (nChannels) + idxChan] = spot.getFeature('MEAN_INTENSITY%02d' % (iChan+1))
						idxChan += 1
				idxSpot += 1

			trackPic = ImagePlus("slice", trackPic)
			# padding
			IJ.run(trackPic, "Canvas Size...", "width="+str(embSize)+" height="+str(embSize)+" position=Center zero")
			trackPic = trackPic.getProcessor()
			embedding.addSlice(str(sliceNum), trackPic)
			sliceNum += 1
	# linear interpolation
	embedding = ImagePlus("embedding", embedding)
	IJ.run(embedding, "Size...", "width="+str(embSize*2)+" height="+str(embSize*2)+" depth="+str(sliceNum+1)+" constrain average interpolation=Bilinear")
	outputFile.replace('.csv', '_mimicry_embd.tif')
	IJ.saveAs(embedding, "Tiff", outputFile);
	embedding.close()
	return None

def writeMeasurements(model,outputFile,nChannels):
	IJ.log('ZedMate completed successfully.' )
	IJ.log( 'Found %d spots in %d tracks.' % ( model.getSpots().getNSpots( True ) , model.getTrackModel().nTracks( True ) ) )

	# Print results in the console.
	headerStr = '%10s %10s %10s %10s %10s' % ( 'Spot_ID', 'Track_ID', 'ZFrame', 'X', 'Y')
	rowStr = '%10d %10d %10d %10.1f %10.1f'
	
	for i in range( nChannels ):
		headerStr += ( ' %10s'  % ( 'C' + str(i+1) ) )
		rowStr += ( ' %10.1f' )
	with open(outputFile, "w") as text_file:
		IJ.log('\n')
		IJ.log( headerStr)
		text_file.write(headerStr+'\n')
		
		tm = model.getTrackModel()		

		#results = compileSpots(tm)
		results = compileTracks(tm, nChannels, rowStr)
		
		for line in results:
			text_file.write(line+"\n")

def runGUI(defaultTargetChannel=2, defaultdt = 1.0, defaultRadius = 0.3, defaultThreshold = 16, defaultFrameGap = 0.01, defaultLinkingMax = 0.01, defaultClosingMax = 0.01):
	gd = NonBlockingGenericDialog("ZedMate - v0.18 beta")
	gd.addMessage("\tZedMate is a TrackMate-based 3D prticle analyzer \n\t\t\t\t\t\t\t\t\t\t\t(copyright Artur Yakimovich 2018-19)\n\n")
	gd.addStringField("File_extension", ".tif")
	gd.addStringField("File_name_contains", "")
	gd.addNumericField("Target_Channel", defaultTargetChannel, 0)
	gd.addNumericField("dt", defaultdt, 2)
	gd.addNumericField("Radius", defaultRadius, 2)
	gd.addNumericField("Threshold", defaultThreshold, 2)
	gd.addNumericField("Frame_Gap", defaultFrameGap, 0)
	gd.addNumericField("Linking_Max", defaultLinkingMax, 2)
	gd.addNumericField("Closing_Max", defaultClosingMax, 2)
	gd.addMessage("\t\t\t\t\t\t_______________________________________")
	gd.addCheckbox("Preview Parameters on the First Image Only", 0)
	gd.addMessage("\t\t\t\t\t(Doesn't save results. Re-opens this Dialog).")
	gd.addMessage("\t\t\t\t\t\t_______________________________________")
	gd.addCheckbox("Save MNIST mimicry embedding (beta)", 0)
	gd.showDialog()

	if gd.wasCanceled():
		return
	extension = gd.getNextString()
	containString = gd.getNextString()
	targetChannel = int(gd.getNextNumber())
	dt = gd.getNextNumber()
	radius = gd.getNextNumber()
	threshold = gd.getNextNumber()
	frameGap = int(gd.getNextNumber())
	linkingMax = gd.getNextNumber()
	closingMax = gd.getNextNumber()
	testMode = gd.getNextBoolean()
	mimicryEmbd = gd.getNextBoolean()

	inputDir = IJ.getDirectory("Input_directory")
	if not inputDir:
		return
	if not testMode:
		outputDir = IJ.getDirectory("Output_directory")
		if not outputDir:
			return
	else:
		outputDir = inputDir # for the case of test
	
	#if not os.path.exists(outputDir):
	#	os.makedirs(outputDir)

	runBatch(inputDir, outputDir, extension, containString, targetChannel, dt, radius, threshold, frameGap,\
			 linkingMax, closingMax, testMode, mimicryEmbd)

  
runGUI()
IJ.log("ZedMate finished processing files")