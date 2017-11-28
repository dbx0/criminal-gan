#!/usr/bin/env bash

#Checking if convert and mogrify are installed
command -v convert >/dev/null 2>&1 || { echo >&2 "I require convert but it's not installed.  Aborting."; exit 1; }
command -v mogrify >/dev/null 2>&1 || { echo >&2 "I require mogrify but it's not installed.  Aborting."; exit 1; }

cd ../data/
wget https://s3.amazonaws.com/nist-srd/SD18/sd18.zip &>/dev/null
wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip &>/dev/null
wget ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip &>/dev/null
wget http://www-prima.inrialpes.fr/perso/Gourier/Faces/HeadPoseImageDatabase.tar.gz &>/dev/null

mkdir all_images

unzip sd18.zip &>/dev/null
mv sd18/*/*/*_F.png all_images/ &>/dev/null
rm -rf sd18 &>/dev/null

unzip BioID-FaceDatabase-V1.2.zip -d bioid/ &>/dev/null
convert bioid/*.pgm all_images/bioid_%03d.png &>/dev/null
rm -rf bioid/ &>/dev/null

unzip att_faces.zip -d att/ &>/dev/null
convert att/*/*.pgm all_images/att_%03d.png &>/dev/null
rm -rf att/ &>/dev/null

tar xvz -f HeadPoseImageDatabase.tar.gz Front/ &>/dev/null
convert Front/*.jpg all_images/att_%03d.png &>/dev/null 
rm -rf Front/ &>/dev/null 

mkdir resized/ &>/dev/null
mogrify -resize 64x64! -quality 100 -type Grayscale -path resized/ all_images/* &>/dev/null
rm -rf all_images/ &>/dev/null

mkdir datasets
mv *zip datasets/
mv *tar.gz datasets/
