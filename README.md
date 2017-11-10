# Criminal-gan

A GAN created to generate faces based on records of criminals

***NIST Special Database 18***

*NIST Mugshot Identification Database (MID)*

*https://www.nist.gov/srd/nist-special-database-18*

## Requirements
* Linux or Mac
* Python 3.6
* Pip

## Instalation
Get into the project folder

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

Downloading dataset 
```bash
wget https://s3.amazonaws.com/nist-srd/SD18/sd18.zip
unzip sd18.zip
```

Move all the images with frontal face to somewhere inside the data folder
```bash
mv sd18/*/*/*_F.png data/front/
```

Install in your system the package imagemagick and use mogrify to resize all images
```bash
mogrify -resize 64x64! -quality 100 -path resized/ *.png
```

## Start 

```bash
python gan.py
```


## Example

**Input with real images**


<img src="https://i.imgur.com/BrtvDmt.png" width="500">

**Output generated with 25 epochs**


<img src="https://i.imgur.com/vbOhdSo.png" width="500">

