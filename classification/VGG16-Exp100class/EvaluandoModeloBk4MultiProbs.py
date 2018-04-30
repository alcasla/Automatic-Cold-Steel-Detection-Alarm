import os
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc		#image reading
from fpdf import FPDF		#generatorPDF


train_data_dir = 'BK4'
test_data_dir = 'Test'
noWeapon_data_samples 	= 252
weapon_data_samples 	= 260
#Define image dimensions, unified
img_width, img_height = 160, 120

#Read json content file, model architecture
json_file = open('modelBK4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#Load arquitecture model readed
loaded_model = model_from_json(loaded_model_json)

#Load weights to model readed
loaded_model.load_weights("BK4.h5")
print('Model generated')

loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#Read transformed imaged by convolution layer blocks
test_data = np.load(open('convolutionedTestImages.npy','rb'))
print("Test data: " + repr(test_data.shape))
nb_test_samples = test_data.shape[0]
print("Test samples: " + repr(nb_test_samples))

#Test directory reading
datagen = ImageDataGenerator(rescale=1./255, data_format='channels_first')
generatorTest = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

#Prepare labels
labels = []
i = 0
for _, y in generatorTest:
    i += len(y)
    labels.append(y)
    if i == nb_test_samples:
        break
labels = np.concatenate(labels)
test_labels = labels 
print("Labels shape: " + repr(test_labels.shape))
# np.set_printoptions(threshold=np.inf)		#print all labels
# print(test_labels)
# for t in range(0,test_labels.shape[1]) :	#print number of samples from each class
# 	print("Label " + repr(t) + " :" + repr(sum(test_labels[:,t])))


### PREDICTION
m = loaded_model.predict_classes([test_data], batch_size=1)
mProb = loaded_model.predict_proba([test_data], batch_size=1)
print(m.shape)

#Split prediction depending on WEAPON or NO WEAPON class
# No weapons
noWeapon_pre = m[:noWeapon_data_samples]
print(noWeapon_pre)
print(noWeapon_pre.shape)
noWeapon_pre_probs = mProb[:noWeapon_data_samples,:]
# Weapons
weapon_pre = m[noWeapon_data_samples:]
print(weapon_pre)
print(weapon_pre.shape)
weapon_pre_probs = mProb[noWeapon_data_samples:,:]


#Class WEAPON image file names
mypathP='Test/Pistolas'		#Path to test images
onlyfilesP = [ f for f in listdir(mypathP) if isfile(join(mypathP,f)) ]
#print(onlyfilesP)
print("Files in weapon directory: " + repr(len(onlyfilesP)))

total_weapons = weapon_data_samples

#Class NoWeapon image file names
mypathN='Test/NoArma'		#Path to test images
onlyfilesN = [ f for f in listdir(mypathN) if isfile(join(mypathN,f)) ]
#print(onlyfilesN)
print("Files in no weapon directory: " + repr(len(onlyfilesN)) + "\n")

total_noWeapons = noWeapon_data_samples




#### Statistics
TP=sum(weapon_pre==0)		#Weapons with right prediction
FP=sum(noWeapon_pre==0)		#NoWeapon with weapon class prediction

print('Total Clase Pistola Real')
print(total_weapons)
print('Total Clase Pistola Aciertos')
print(TP)
print('Total Clase NoArma Real')
print(total_noWeapons)
print ('Total Clase NoArma Errores')
print(FP)

Acc = float(TP)/(TP+FP)
print('\nPrecision: ' + repr(Acc))

FN = total_weapons-TP
Rec = float(TP)/(TP+FN)
print('Recall: ' + repr(Rec))

F1m = ( (Acc*Rec)/(Acc+Rec) )*2
print('F1means: ' + repr(F1m))

print("\n\n")


### THRESHOLD, predictions overhead the threshold value = accurate predictions TERMINAL
threshold = 0.5
while threshold <= 1.01 :		#incremental threshold
	#TP and FP above the threshold
	thTP = sum(weapon_pre_probs[:,0] >= threshold)
	thFP = sum(noWeapon_pre_probs[:,0] >= threshold)
	print("Threshold: " + repr(threshold) + "  -  TP: " + repr(thTP) + " - FP: " + repr(thFP)),
	#quality measures depens on the threshold
	thAcc = round(float(thTP)/(thTP+thFP), 3)
	thRec = round(float(thTP)/total_weapons, 3)
	thF1m = round(( (thAcc*thRec)/(thAcc+thRec) )*2, 3)
	print("--- Acc = " + repr(thAcc) + " - Rec = " + repr(thRec) + " -- F1 = " + repr(thF1m))
	threshold += 0.05
	threshold = round(threshold, 2)
print("\n\n")





##########################################################################
#### REPORTER PDF ####
def ReporterPDF():
	#Get class names with index to identify prediction probs
	datagen = ImageDataGenerator(rescale=1./255, data_format='channels_first')
	generator = datagen.flow_from_directory(
	    train_data_dir,
	    target_size=(img_width, img_height),
	    batch_size=32,
	    class_mode='categorical',
	    shuffle=False)
	classes = generator.class_indices.items()

	classNames = ["" for x in range(100)]
	i=0
	for i in range(0,100):
		classNames[classes[i][1]] = classes[i][0]
	#print(classNames)


	#### ReporterPDF ####
	print('\nReporterPDF working ...')

	pdf = FPDF('P', 'mm', 'A4')
	pdf.add_page()
	pdf.set_font('Arial', 'B', 9)

	### Content

	### THRESHOLD, predictions overhead the threshold value = accurate predictions ReporterPDF
	pdf.cell(50, 10, 'Estadisticos', 1, 0, 'C')
	pdf.ln(16)

	threshold = 0.5
	while threshold <= 1.01 :		#incremental threshold
		#TP and FP above the threshold
		thTP = sum(weapon_pre_probs[:,0] >= threshold)
		thFP = sum(noWeapon_pre_probs[:,0] >= threshold)
		#quality measures depens on the threshold
		thAcc = round(float(thTP)/(thTP+thFP), 3)
		thRec = round(float(thTP)/total_weapons, 3)
		thF1m = round(( (thAcc*thRec)/(thAcc+thRec) )*2, 3)

		pdf.cell(60, 8, "Threshold: " + repr(threshold) + "  -  TP: " + repr(thTP) + " - FP: " + repr(thFP))	#dump statistics
		pdf.cell(100, 8, "--- Acc = " + repr(thAcc) + " - Rec = " + repr(thRec) + " -- F1 = " + repr(thF1m))
		pdf.ln(8)

		threshold += 0.05		#update threshold
		threshold = round(threshold, 2)


	pdf.add_page()	#separate statistics from images


	#KNIFES WRONG
	pdf.cell(50, 10, 'KNIFES mal clasificadas', 1, 0, 'C')
	pdf.ln(11)

	for i in range(0, len(onlyfilesP)) :
		#insert ranking probs and image
		if weapon_pre[i] != 0:
			roundedProbs = [ '%.5f' % elem for elem in weapon_pre_probs[i,:] ]	#Get 100 image probs and round
			rankProbs = sorted(roundedProbs, reverse=True)						#order probs in descending order
			for r in rankProbs[0:3] :											#print three cells for highest probs
				if r != 0 :
					pdf.cell( 55, 8, str(classNames[roundedProbs.index(r)] + ': ' + repr(r)), 1, 0, 'C' )
			pdf.ln(9)
			pdf.image('Test/Pistolas/'+ onlyfilesP[i], w=44, h=33)				#draw image
			pdf.ln(0.5)


	pdf.add_page()	#separate different class images

	# #NoKNIFES WRONG
	pdf.cell(50, 10, 'NoKnifes mal clasificadas', 1, 0, 'C')
	pdf.ln(11)

	for j in range(0, len(onlyfilesN)):
		#insert ranking probs and image
		if noWeapon_pre[j] == 0:
			roundedProbs = [ '%.5f' % elem for elem in noWeapon_pre_probs[j,:] ]	#Get 100 image probs and round
			rankProbs = sorted(roundedProbs, reverse=True)							#order probs in descending order
			for r in rankProbs[0:3] :
				if r != 0 :
					pdf.cell( 55, 8, str(classNames[roundedProbs.index(r)] + ': ' + repr(r)), 1, 0, 'C' )
			pdf.ln(9)
			pdf.image('Test/NoArma/'+ onlyfilesN[j], w=44, h=33)
			pdf.ln(0.5)

	pdf.add_page()  #separate different class images

	#KNIFES RIGHT
	pdf.cell(50, 10, 'KNIFES bien clasificadas', 1, 0, 'C')
	pdf.ln(11)

	for i in range(0, len(onlyfilesP)):
		#insert ranking probs and image
		if weapon_pre[i] == 0:
			roundedProbs = [ '%.5f' % elem for elem in weapon_pre_probs[i,:] ]	#Get 100 image probs and round
			rankProbs = sorted(roundedProbs, reverse=True)						#order probs in descending order
			for r in rankProbs[0:3] :
				if r != 0 :
					pdf.cell( 55, 8, str(classNames[roundedProbs.index(r)] + ': ' + repr(r)), 1, 0, 'C' )
			pdf.ln(9)
			pdf.image('Test/Pistolas/'+ onlyfilesP[i], w=44, h=33)
			pdf.ln(0.5)

	pdf.add_page()  #separate different class images

	#NoKNIFES RIGHT
	pdf.cell(50, 10, 'NoKnifes bien clasificadas', 1, 0, 'C')
	pdf.ln(11)

	for j in range(0, len(onlyfilesN)):
		#insert ranking probs and image
		if noWeapon_pre[j] != 0:
			roundedProbs = [ '%.5f' % elem for elem in noWeapon_pre_probs[j,:] ]	#Get 100 image probs and round
			rankProbs = sorted(roundedProbs, reverse=True)							#order probs in descending order
			for r in rankProbs[0:3] :
				if r != 0 :
					pdf.cell( 55, 8, str(classNames[roundedProbs.index(r)] + ': ' + repr(r)), 1, 0, 'C' )
			pdf.ln(9)
			pdf.image('Test/NoArma/'+ onlyfilesN[j], w=44, h=33)
			pdf.ln(0.5)

	### end content

	pdf.output('report.pdf', 'F')
	print('Report: succesful generation\n')




####### Main execution #######
ReporterPDF()
