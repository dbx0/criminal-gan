#!/usr/bin/env bash

echo "Dataset installer"
echo "Status: "
echo -ne "Verifying command dependencies...  [                 ](0%)\r"
sleep 1
#Checking if convert and mogrify are installed
command -v convert >/dev/null 2>&1 || { echo >&2 "I require convert but it's not installed.  Aborting."; exit 1; }
command -v mogrify >/dev/null 2>&1 || { echo >&2 "I require mogrify but it's not installed.  Aborting."; exit 1; }
sleep 1

echo -ne "Downloading Datasets...  [|                ](5%)\r"
sleep 1
cd ../data/
#downloading all the dataset
#wget https://s3.amazonaws.com/nist-srd/SD18/sd18.zip &>/dev/null
echo -ne "Downloading Datasets...  [||               ](15%)\r"
sleep 1
#wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip &>/dev/null
echo -ne "Downloading Datasets...  [|||              ](20%)\r"
sleep 1
#wget ftp://ftp.uni-erlangen.de/pub/facedb/BioID-FaceDatabase-V1.2.zip &>/dev/null
echo -ne "Downloading Datasets...  [||||             ](25%)\r"
sleep 1
#wget http://www-prima.inrialpes.fr/perso/Gourier/Faces/HeadPoseImageDatabase.tar.gz &>/dev/null
echo -ne "Downloading Datasets...  [|||||            ](30%)\r"
sleep 1

sleep 1

mkdir all_images
sleep 1

echo -ne "Setting up SD18...  [||||||           ](35%)\r"
sleep 1
#unziping and setting up the images 
unzip sd18.zip &>/dev/null
echo -ne "Setting up SD18...  [|||||||          ](40%)\r"
sleep 1
mv sd18/*/*/*_F.png all_images/ &>/dev/null
echo -ne "Setting up SD18...  [||||||||         ](45%)\r"
sleep 1
rm -rf sd18 &>/dev/null
sleep 1

echo -ne "Setting up BioID...  [||||||||         ](45%)\r"
sleep 1
unzip BioID-FaceDatabase-V1.2.zip -d bioid/ &>/dev/null
echo -ne "Setting up BioID...  [|||||||||        ](50%)\r"
sleep 1
convert bioid/*.pgm all_images/bioid_%03d.png &>/dev/null
echo -ne "Setting up BioID...  [||||||||||       ](55%)\r"
sleep 1
rm -rf bioid/ &>/dev/null
sleep 1

echo -ne "Setting up AT&T DS...  [||||||||||       ](55%)\r"
sleep 1
unzip att_faces.zip -d att/ &>/dev/null
echo -ne "Setting up AT&T DS...  [|||||||||||       ](60%)\r"
sleep 1
convert att/*/*.pgm all_images/att_%03d.png &>/dev/null
echo -ne "Setting up AT&T DS...  [||||||||||||     ](65%)\r"
sleep 1
rm -rf att/ &>/dev/null
sleep 1

echo -ne "Setting up HPID...  [||||||||||||     ](65%)\r"
sleep 1
tar xvz -f HeadPoseImageDatabase.tar.gz Front/ &>/dev/null
echo -ne "Setting up HPID...  [|||||||||||||    ](70%)\r"
sleep 1
convert Front/*.jpg all_images/att_%03d.png &>/dev/null 
echo -ne "Setting up HPID...  [||||||||||||||   ](75%)\r"
sleep 1
rm -rf Front/ &>/dev/null 
sleep 1

echo -ne "Converting images...  [||||||||||||||   ](75%)\r"
sleep 1
mkdir resized/ &>/dev/null
echo -ne "Converting images...  [|||||||||||||||  ](80%)\r"
sleep 1
mogrify -resize 64x64! -quality 100 -type Grayscale -path resized/ all_images/* &>/dev/null
echo -ne "Converting images...  [|||||||||||||||| ](95%)\r"
sleep 1
rm -rf all_images/ &>/dev/null
sleep 1

echo "Finished."
sleep 5