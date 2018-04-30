import os
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from os import listdir
from os.path import isfile, join
import numpy
from scipy import misc		#image reading
from fpdf import FPDF


#Define image dimensions, unified
img_width, img_height = 160, 120

#Read json content file, model architecture
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#Load arquitecture model readed
loaded_model = model_from_json(loaded_model_json)

#Load weights
loaded_model.load_weights("ModeloPr.h5")
print("Loaded model")

#define train parameters
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Class Arma
mypathP='path_to_weapon_folder'		#test for weapon
onlyfilesP = [ f for f in listdir(mypathP) if isfile(join(mypathP,f)) ]
print onlyfilesP

#create multidimensional array
imagenes = numpy.empty((len(onlyfilesP),160,120,3),dtype=int)

#fill array with resized images
n=0
for n in range(0, len(onlyfilesP)) :
	print onlyfilesP[n]
	imagen = (misc.imread( join(mypathP,onlyfilesP[n]) )).astype(numpy.float32)
	if len(imagen.shape) == 3:
		imagen = imagen[: , : , 0:3]
	imagen = imagen.transpose((1,0,2))
	imagen = misc.imresize(imagen, (160,120,3), 'bilinear')
	print imagen.shape
	imagenes[n]=imagen
Total1=n+1

m=loaded_model.predict_classes([imagenes], batch_size=1)
print m
print m.shape

#Class NoArma
mypathN='path_to_not_weapon_folder'		#test for not weapon
onlyfilesN = [ f for f in listdir(mypathN) if isfile(join(mypathN,f)) ]
print onlyfilesN

imagenes2 = numpy.empty((len(onlyfilesN),160,120,3),dtype=int)
n=0
for n in range(0, len(onlyfilesN)) :
	print onlyfilesN[n]
	imagen = (misc.imread( join(mypathN,onlyfilesN[n]) )).astype(numpy.float32)
	if len(imagen.shape) == 3:
		imagen = imagen[: , : , 0:3]
	imagen = imagen.transpose((1,0,2))
	imagen = misc.imresize(imagen, (160,120,3), 'bilinear')
	print imagen.shape
	imagenes2[n]=imagen
Total2=n+1
m2=loaded_model.predict_classes([imagenes2], batch_size=1)
print m2
print m2.shape


#### Wrong classified imagenes - print names each class
print 'Armas erroneas'
i=0
for i in range(0, len(onlyfilesP)):
	if m[i] == 0:
		print onlyfilesP[i]

print '\n\nNoArmas erroneas'
j=0
for j in range(0, len(onlyfilesN)):
        if m2[j] == 1:
                print onlyfilesN[j]
print '\n\n'



#### Statistics
print ("Total Clase Pistola Real")
print(Total1)
print ("Total Clase Pistola Aciertos")
TP=sum(m)[0]
print(TP)
print ("Total Clase NoArma Real")
print(Total2)
print ("Total Clase NoArma Errores") 
FP=sum(m2)[0]
print(FP)

Acc = float(TP)/(TP+FP)
print "\nPrecision: " + repr(Acc)

FN = Total1-TP
Rec = float(TP)/(TP+FN)
print "Recall: " + repr(Rec)

F1m = ( (Acc*Rec)/(Acc+Rec) )*2
print "F1means: " + repr(F1m)





#### ReporterPDF ####
#Generate pdf with grouped images by TP, FP, TN, FN from test
#####################
print "\n\nReporterPDF working ..."
pdf = FPDF('P', 'mm', 'A4')
pdf.add_page()
pdf.set_font('Arial', 'B', 11)

### content
#Knifes wrong
pdf.cell(50, 10, 'KNIFES mal clasificadas', 1, 0, 'C')
pdf.ln(11)

xImg = 6	#variables to locate images via incremnts
yImg = 22
contX=0		#counters of images and lines added
contY=0
i=0		#counter to list all test images
for i in range(0, len(onlyfilesP)):
        if m[i] == 0:
		pdf.image('Test/Pistolas/'+ onlyfilesP[i], x=xImg, y=yImg, w=48, h=36)
		xImg += 50	#Jump image width plus space to next image
		contX += 1
		if (contX % 4) == 0:	#new line
			xImg = 6
			yImg += 38
			contY += 1
			if (contY % 7) == 0:	#new page
				yImg = 22
				pdf.add_page()

pdf.add_page()	#separate different class images

#Pistols worng
pdf.cell(50, 10, 'PISTOLS mal clasificadas', 1, 0, 'C')
pdf.ln(11)

xImg = 6        #variables to locate images via incremnts
yImg = 22
contX=0         #counters of images and lines added
contY=0
j=0             #counter to list all test images
for j in range(0, len(onlyfilesN)):
        if m2[j] == 1:
                pdf.image('Test/NoArma/'+ onlyfilesN[j], x=xImg, y=yImg, w=48, h=36)
                xImg += 50      #Jump image width plus space to next image
                contX += 1
                if (contX % 4) == 0:    #new line
                        xImg = 6
                        yImg += 38
                        contY += 1
                        if (contY % 7) == 0:    #new page
                                yImg = 22
                                pdf.add_page()

pdf.add_page()  #separate different class images

#Knifes right
pdf.cell(50, 10, 'KNIFES bien clasificadas', 1, 0, 'C')
pdf.ln(11)

xImg = 6        #variables to locate images via incremnts
yImg = 22
contX=0         #counters of images and lines added
contY=0
i=0             #counter to list all test images
for i in range(0, len(onlyfilesP)):
        if m[i] == 1:
                pdf.image('Test/Pistolas/'+ onlyfilesP[i], x=xImg, y=yImg, w=48, h=36)
                xImg += 50      #Jump image width plus space to next image
                contX += 1
                if (contX % 4) == 0:    #new line
                        xImg = 6
                        yImg += 38
                        contY += 1
                        if (contY % 7) == 0:    #new page
                                yImg = 22
                                pdf.add_page()

pdf.add_page()  #separate different class images

#Pistols right
pdf.cell(50, 10, 'PISTOLS bien clasificadas', 1, 0, 'C')
pdf.ln(11)

xImg = 6        #variables to locate images via incremnts
yImg = 22
contX=0         #counters of images and lines added
contY=0
j=0             #counter to list all test images
for j in range(0, len(onlyfilesN)):
        if m2[j] == 0:
                pdf.image('Test/NoArma/'+ onlyfilesN[j], x=xImg, y=yImg, w=48, h=36)
                xImg += 50      #Jump image width plus space to next image
                contX += 1
                if (contX % 4) == 0:    #new line
                        xImg = 6
                        yImg += 38
                        contY += 1
                        if (contY % 7) == 0:    #new page
                                yImg = 22
                                pdf.add_page()
### end content

pdf.output('report.pdf', 'F')
print "Report: succesful generation\n"
