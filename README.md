# Criminal-gan

A GAN created to generate faces based on records of criminals

## Requirements
* Linux or Mac
* Python 3.6
* Pip
* imagemagick

## Datasets

In this case we are going to use the following datasets:

***NIST Special Database 18***

*NIST Mugshot Identification Database (MID)*

*https://www.nist.gov/srd/nist-special-database-18*

***AT&T Laboratories Cambridge hosted in conjunction with Cambridge University Computer Laboratory***

*The Database of Faces*

*http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html*

***BioID Face Database - FaceDB***

*https://www.bioid.com/About/BioID-Face-Database*

***FGnet - IST-2000-26434 Face and Gesture Recognition Working group***

*Head Pose Image Database*

*http://www-prima.inrialpes.fr/FGnet/html/home.html*


## Project instalation
Clone this project and run the following

Setting up the virtual env
```bash
pip install virtualenv
virtualenv venv_cgan
source venv_cgan/bin/activate
```

Installing dependencies
```bash
pip install -r requirements.txt
```

## Setting up datasets

Download the datasets
```bash
cd data/
wget https://s3.amazonaws.com/nist-srd/SD18/sd18.zip
wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
wget ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip
wget http://www-prima.inrialpes.fr/perso/Gourier/Faces/HeadPoseImageDatabase.tar.gz
```

Create a folder to contain all the dataset
```bash
mkdir all_images
```

#### Unzip the folders, convert if necessary and move to a single folder


**NIST Special Database 18**
```bash
unzip sd18.zip
mv sd18/*/*/*_F.png all_images/
rm -rf sd18
```

**BioID Face Database - FaceDB**
```bash
unzip BioID-FaceDatabase-V1.2.zip -d bioid/
convert bioid/*.pgm all_images/bioid_%03d.png
rm -rf bioid/
```

**AT&T - The Database of Faces**
```bash
unzip att_faces.zip -d att/
convert att/*/*.pgm all_images/att_%03d.png
rm -rf att/
```

**FGnet - Head Pose Image Database**
```bash
tar xvz -f HeadPoseImageDatabase.tar.gz Front/
convert Front/*.jpg all_images/att_%03d.png
rm -rf Front/
```

```bash
mkdir resized/
mogrify -resize 64x64! -quality 100 -path resized/ all_images/* -type Grayscale
rm -rf all_images/
```

## Start 

```bash
$ python gan.py


	+-+-+-+-+-+-+-+-+ +-+-+-+
	|C|R|I|M|I|N|A|L| |G|A|N|
	+-+-+-+-+-+-+-+-+ +-+-+-+
	Created by Davidson Mizael
	


# Starting...
# Loading data...
# Starting generator and descriminator...
# Starting epochs (15)...
# Progress: [0/15][38/38] Loss_D: 1.5640 Loss_G: 0.0004
# Progress: [1/15][38/38] Loss_D: 1.4570 Loss_G: 0.0004
...
```