#!/bin/bash 

mkdir -p videos
mkdir -p first_frame
mkdir -p output

cd videos
base='https://hiring.verkada.com/video/'
end='.ts'
url="$base$1$end"
curl -s "$url" --output $1.ts
cd ..

echo "downloading $url..."

path='./videos/'
path_to_vid="$path$1$end"

path_to_image='./first_frame/'

ffmpeg -hide_banner -loglevel warning -ss 00:00:00 -i $path_to_vid -vframes 1 -q:v 2 $path_to_image$1.jpg

echo "extracting image... wrote $1.jpg"
